#!/usr/bin/env python3

# Precompute mel-spectrogram features for KaldiDiarizationDataset.
#
# Rationale:
#   - get_labeledSTFT() reads audio for a specific chunk (start/end frames)
#     and STFTs *that* segment. If you instead STFT/mel the whole recording
#     and slice afterwards, framing (window boundaries) differs from the
#     per-chunk computation done at train time -> different features.
#     So chunk boundaries must be decided first, exactly like
#     KaldiDiarizationDataset.__init__ does, and only THEN do we STFT+mel
#     each chunk.
#   - We deliberately stop right after features.transform() (the mel
#     spectrogram). splice() / subsample() / specaugment / top-N speaker
#     selection are all cheap, and depend on hyperparameters
#     (context_size, subsampling) you may still want to change later,
#     so they are done at load time in PrecomputedKaldiDiarizationDataset,
#     not here.
#   - Chunk boundaries are computed with subsampling fixed to 1, so the
#     precomputed chunks are independent of the subsampling value you pick
#     later (subsampling only affects the loop, not the underlying audio
#     span, and we don't want to have to re-precompute if you change it).

import argparse
import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool

import numpy as np

import common_utils.features as features
import common_utils.kaldi_data as kaldi_data


def _count_frames(init: int, data_len: int, size: int, step: int) -> int:
    return int((init + data_len - size + step) / step)


def _gen_frame_indices(
    init_frame: int,
    data_length: int,
    size: int,
    step: int,
    use_last_samples: bool,
    min_length: int,
):
    i = -1
    for i in range(_count_frames(init_frame, data_length, size, step)):
        yield init_frame + (i * step), init_frame + (i * step) + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (init_frame + (i + 1) * step) > min_length:
            yield init_frame + (i + 1) * step, data_length


def build_chunk_indices(
    data: kaldi_data.KaldiData,
    chunk_size: int,
    sampling_rate: int,
    frame_shift: int,
    use_last_samples: bool,
    min_length: int,
):
    """Same logic as KaldiDiarizationDataset.__init__, with subsampling
    fixed to 1 (subsampling is applied at load time instead)."""
    chunk_indices = []
    for rec in data.wavs:
        data_len = int(data.reco2dur[rec] * sampling_rate / frame_shift)
        if data.uem:
            init_frame = int(data.uem[rec][0] * sampling_rate / frame_shift)
            data_len = int(data.uem[rec][1] * sampling_rate / frame_shift)
        else:
            init_frame = 0
        if chunk_size > 0:
            for st, ed in _gen_frame_indices(
                init_frame, data_len, chunk_size, chunk_size,
                use_last_samples, min_length,
            ):
                chunk_indices.append((rec, st, ed))
        else:
            chunk_indices.append((rec, 0, data_len))
    return chunk_indices


# KaldiData is not picklable-friendly / cheap to share across processes
# (it holds open globs, etc.), so each worker builds its own instance,
# cached per-process.
_worker_data_cache = {}


def _get_worker_data(data_dir: str) -> kaldi_data.KaldiData:
    if data_dir not in _worker_data_cache:
        _worker_data_cache[data_dir] = kaldi_data.KaldiData(data_dir)
    return _worker_data_cache[data_dir]


def process_one(
    idx_chunk,
    data_dir: str,
    frame_size: int,
    frame_shift: int,
    n_speakers,
    sampling_rate: int,
    feature_dim: int,
    input_transform: str,
    output_dir: str,
):
    idx, (rec, st, ed) = idx_chunk
    data = _get_worker_data(data_dir)

    Y, T, speaker_ids = features.get_labeledSTFT(
        data, rec, st, ed, frame_size, frame_shift, n_speakers)

    if Y.shape[0] == 0:
        Y_t = np.zeros((0, feature_dim), dtype=np.float32)
    else:
        # specaugment is a training-time random augmentation - never
        # bake it into the precomputed cache, it must stay dynamic.
        Y_t = features.transform(
            Y, sampling_rate, feature_dim, input_transform,
            specaugment=False)

    out_path = os.path.join(output_dir, f"{idx:08d}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(
            {'Y': Y_t, 'T': T, 'rec': rec, 'st': st, 'ed': ed,
             'speaker_ids': speaker_ids},
            f, protocol=pickle.HIGHEST_PROTOCOL)
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Precompute mel-spectrogram features for "
                     "KaldiDiarizationDataset (stops right after "
                     "features.transform(), before splice/subsample).")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)

    # Same arg surface as KaldiDiarizationDataset, so you can reuse your
    # existing config/CLI setup. Args noted "(load-time)" are accepted
    # here only for symmetry/documentation and are NOT used during
    # precompute; pass them again to PrecomputedKaldiDiarizationDataset.
    parser.add_argument('--chunk-size', type=int, default=2000)
    parser.add_argument('--context-size', type=int, default=0,
                         help="(load-time) unused here")
    parser.add_argument('--feature-dim', type=int, default=23)
    parser.add_argument('--frame-shift', type=int, default=80)
    parser.add_argument('--frame-size', type=int, default=200)
    parser.add_argument('--input-transform', type=str,
                         default='logmel23_meannorm')
    parser.add_argument('--n-speakers', type=int, default=None)
    parser.add_argument('--sampling-rate', type=int, default=8000)
    parser.add_argument('--shuffle', action='store_true',
                         help="(load-time) unused here")
    parser.add_argument('--subsampling', type=int, default=1,
                         help="(load-time) unused here")
    parser.add_argument('--use-last-samples', action='store_true')
    parser.add_argument('--min-length', type=int, default=0)
    parser.add_argument('--specaugment', action='store_true',
                         help="(load-time) ignored here; specaugment is "
                              "never precomputed, always applied fresh "
                              "at load time")
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s %(levelname)s %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    data = kaldi_data.KaldiData(args.data_dir)
    chunk_indices = build_chunk_indices(
        data, args.chunk_size, args.sampling_rate, args.frame_shift,
        args.use_last_samples, args.min_length)

    logging.info(f"#files: {len(data.wavs)}, #chunks: {len(chunk_indices)}")

    meta = {
        'data_dir': args.data_dir,
        'chunk_size': args.chunk_size,
        'feature_dim': args.feature_dim,
        'frame_shift': args.frame_shift,
        'frame_size': args.frame_size,
        'input_transform': args.input_transform,
        'n_speakers': args.n_speakers,
        'sampling_rate': args.sampling_rate,
        'use_last_samples': args.use_last_samples,
        'min_length': args.min_length,
        'chunk_indices': chunk_indices,
    }
    with open(os.path.join(args.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    worker_fn = partial(
        process_one,
        data_dir=args.data_dir,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        n_speakers=args.n_speakers,
        sampling_rate=args.sampling_rate,
        feature_dim=args.feature_dim,
        input_transform=args.input_transform,
        output_dir=args.output_dir,
    )

    items = list(enumerate(chunk_indices))
    total = len(items)

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(worker_fn, items), 1):
                if done % 200 == 0 or done == total:
                    logging.info(f"Processed {done}/{total}")
    else:
        for done, item in enumerate(items, 1):
            worker_fn(item)
            if done % 200 == 0 or done == total:
                logging.info(f"Processed {done}/{total}")

    logging.info(f"Done. Wrote {total} chunks to {args.output_dir}")


if __name__ == '__main__':
    main()
