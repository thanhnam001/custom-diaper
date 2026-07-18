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
#   - chunk_size here is in RAW frame domain (not subsampled). You choose
#     it directly as the physical audio span you want each chunk to
#     cover, e.g. if you plan to use subsampling=10 and want ~600 frames
#     of output, pass --chunk-size 6000. Because the raw span is fixed at
#     precompute time, subsampling becomes a free load-time knob: you can
#     try subsampling=5, 10, 20, etc. against the same cache without
#     re-running precompute. (The original KaldiDiarizationDataset instead
#     derives raw span as chunk_size * subsampling with chunk_size in the
#     *subsampled* domain -- so if you want your precomputed chunks to
#     exactly reproduce a specific original config's chunking, multiply
#     chunk_size by that config's subsampling yourself before passing it
#     here.)
#     context_size and specaugment are also free to change at load time.

import glob
import argparse
import logging
import io
import os
import pickle
import librosa
from functools import partial, lru_cache
from multiprocessing import Pool
import subprocess
import sys
import numpy as np
from typing import Any
import soundfile as sf
import torch
import torchaudio.transforms as T

# import common_utils.features as features
# import common_utils.kaldi_data as kaldi_data

def load_segments_hash(segments_file):
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        ret[utt] = (rec, float(st), float(et))
    return ret


def load_segments_rechash(segments_file: str) -> dict[str, dict[str, Any]]:
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        if rec not in ret:
            ret[rec] = []
        ret[rec].append({'utt': utt, 'st': float(st), 'et': float(et)})
    return ret


def load_wav_scp(
    wav_scp_file: str, 
    replace_root: tuple[str, str] | None = None,
    base_dir: str | None = None,
    expand_vars: bool = True
) -> dict[str, str]:
    """ 
    Return dictionary { rec: wav_rxfilename }
    
    Args:
        wav_scp_file: Path to wav.scp file
        replace_root: Tuple of (old_root, new_root) to replace paths
        base_dir: Base directory for relative paths. If None, uses wav.scp directory
        expand_vars: Whether to expand environment variables (e.g., $HOME, %USERPROFILE%)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(wav_scp_file))
    
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    
    result = {}
    for rec_id, wav_path in lines:
        # Expand environment variables
        if expand_vars:
            wav_path = os.path.expandvars(os.path.expanduser(wav_path))
        
        # Handle replace_root
        if replace_root is not None:
            wav_path = wav_path.replace(replace_root[0], replace_root[1])
        
        # Normalize path separators to handle mixed separators (e.g., '../a\b')
        # Replace all backslashes with forward slashes first, then let normpath handle it
        if not wav_path.endswith('|') and wav_path != '-':
            wav_path = wav_path.replace('\\', '/')
        
        # Convert relative paths to absolute
        # Skip pipe commands (ending with |) and stdin (-)
        if not wav_path.endswith('|') and wav_path != '-':
            if not os.path.isabs(wav_path):
                # Use normpath to properly resolve '..' and '.' in paths
                # This handles cases like '../a/b' correctly when joined with absolute paths
                wav_path = os.path.normpath(os.path.join(base_dir, wav_path))
            else:
                # Even for absolute paths, normalize to resolve any '..' or '.'
                wav_path = os.path.normpath(wav_path)
        
        result[rec_id] = wav_path
    
    return result


@lru_cache(maxsize=1)
def load_wav(
    wav_rxfilename: str,
    start: int,
    end: int
) -> tuple[np.ndarray, int]:
    """ This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
    """
    if wav_rxfilename.endswith('|'):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1], shell=True,
                             stdout=subprocess.PIPE)
        data, samplerate = sf.read(io.BytesIO(p.stdout.read()),
                                   dtype='float32')
        # cannot seek
        data = data[start:end]
    elif wav_rxfilename == '-':
        # stdin
        data, samplerate = sf.read(sys.stdin, dtype='float32')
        # cannot seek
        data = data[start:end]
    else:
        # normal wav file
        data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
    return data, samplerate


def load_utt2spk(utt2spk_file: str) -> dict[str, str]:
    """ returns dictionary { uttid: spkid } """
    lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
    return {x[0]: x[1] for x in lines}


def load_spk2utt(spk2utt_file: str) -> dict[str, str]:
    """ returns dictionary { spkid: list of uttids } """
    if not os.path.exists(spk2utt_file):
        return None
    lines = [line.strip().split() for line in open(spk2utt_file)]
    return {x[0]: x[1:] for x in lines}


def load_reco2dur(reco2dur_file: str) -> dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(reco2dur_file):
        return None
    lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
    return {x[0]: float(x[1]) for x in lines}


def load_uem(uem_file: str) -> dict[str, tuple[float, float]]:
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split(None) for line in open(uem_file)]
    return {x[0]: (float(x[2]), float(x[3])) for x in lines}

class KaldiData:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.segments = load_segments_rechash(
                os.path.join(self.data_dir, 'segments'))
        self.utt2spk = load_utt2spk(
                os.path.join(self.data_dir, 'utt2spk'))
        self.wavs = load_wav_scp(
                os.path.join(self.data_dir, 'wav.scp'))
        self.reco2dur = load_reco2dur(
                os.path.join(self.data_dir, 'reco2dur'))
        self.spk2utt = load_spk2utt(
                os.path.join(self.data_dir, 'spk2utt'))
        self.uem = load_uem(os.path.join(self.data_dir, 'uem'))

    def load_wav(
        self,
        recid: str,
        start: int,
        end: int
    ) -> tuple[np.ndarray, int]:
        files = glob.glob(self.wavs[recid])
        if len(files) > 0:
            data, rate = load_wav(files[0], start, end)
        else:
            print(f"{self.wavs[recid]} not found")
            data = np.asarray([])
            rate = 0
        return data, rate

def stft(
    data: np.ndarray,
    frame_size: int,
    frame_shift: int
) -> np.ndarray:
    """ Compute STFT features
    Args:
        data: audio signal
            (n_samples,)-shaped np.float32 array
        frame_size: number of samples in a frame (must be a power of two)
        frame_shift: number of samples between frames
    Returns:
        stft: STFT frames
            (n_frames, n_bins)-shaped np.complex64 array
    """
    # round up to nearest power of 2
    fft_size = 1 << (frame_size - 1).bit_length()
    # HACK: The last frame is omitted
    #       as librosa.stft produces such an excessive frame
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T

def get_labeledSTFT(
    kaldi_obj: KaldiData,
    rec: str,
    start: int,
    end: int,
    frame_size: int,
    frame_shift: int,
    n_speakers: int = None,
    use_speaker_id: bool = False
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times
    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex64 array,
        T: label
            (n_frmaes, n_speakers)-shaped np.int32 array.
    """
    data, rate = kaldi_obj.load_wav(
        rec, start * frame_shift, end * frame_shift)
    Y = stft(data, frame_size, frame_shift)
    filtered_segments = kaldi_obj.segments[rec]
    # filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
        [kaldi_obj.utt2spk[seg['utt']] for seg
         in filtered_segments]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)

    all_speakers = sorted(kaldi_obj.spk2utt.keys())
    if use_speaker_id:
        S = np.zeros((Y.shape[0], len(all_speakers)), dtype=np.int32)

    global_spk_indices = []
    for seg in filtered_segments:
        spk = kaldi_obj.utt2spk[seg['utt']]
        speaker_index = speakers.index(spk)
        if not (all_speakers.index(spk) in global_spk_indices):
            global_spk_indices.append(all_speakers.index(spk))
        if use_speaker_id:
            all_speaker_index = all_speakers.index(
                kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
            seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
            seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            if speaker_index < n_speakers:
                T[rel_start:rel_end, speaker_index] = 1
                if use_speaker_id:
                    S[rel_start:rel_end, all_speaker_index] = 1

    if use_speaker_id:
        return Y, T, global_spk_indices, S
    else:
        return Y, T, global_spk_indices


def transform(
    Y: np.ndarray,
    sampling_rate: int,
    feature_dim: int,
    transform_type: str,
    specaugment: bool,
    dtype: type = np.float32,
) -> np.ndarray:
    """ Transform STFT feature
    Args:
        Y: STFT
            (n_frames, n_bins)-shaped array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if transform_type.startswith('logmel'):
        n_fft = 2 * (Y.shape[1] - 1)
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=feature_dim)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        if transform_type == 'logmel_meannorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
        elif transform_type == 'logmel_meanvarnorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
            std = np.maximum(np.std(Y, axis=0), 1e-10)
            Y = Y / std
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    if specaugment:
        timemasking = T.TimeMasking(time_mask_param=80)
        Y = timemasking(torch.from_numpy(Y).unsqueeze(0).transpose(1, 2))
        freqmasking = T.FrequencyMasking(freq_mask_param=80)
        Y = freqmasking(Y.transpose(1, 2)[0]).numpy()
    return Y.astype(dtype)


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
    data: KaldiData,
    chunk_size: int,
    sampling_rate: int,
    frame_shift: int,
    use_last_samples: bool,
    min_length: int,
    tail_subsampling: int = None,
):
    """chunk_size/init_frame/data_len are all in RAW frame domain here
    (subsampling fixed to 1) -- the physical audio span of each chunk is
    exactly chunk_size raw frames, independent of whatever subsampling
    you apply later at load time.

    tail_subsampling: if set, reproduces KaldiDiarizationDataset's
    rounding quirk for a SPECIFIC subsampling value -- it does
    int(data_len / subsampling) then multiplies back by subsampling
    before generating chunks, which silently truncates up to
    (subsampling - 1) raw frames off the end of each recording. This
    only exists so verify_precomputed_dataset.py can bit-for-bit match
    KaldiDiarizationDataset(subsampling=tail_subsampling); it is NOT
    needed for normal precompute use (leave it as None / default)."""
    chunk_indices = []
    for rec in data.wavs:
        data_len = int(data.reco2dur[rec] * sampling_rate / frame_shift)
        if data.uem:
            init_frame = int(data.uem[rec][0] * sampling_rate / frame_shift)
            data_len = int(data.uem[rec][1] * sampling_rate / frame_shift)
        else:
            init_frame = 0

        if tail_subsampling and tail_subsampling > 1:
            data_len = int(data_len / tail_subsampling) * tail_subsampling
            init_frame = int(init_frame / tail_subsampling) * tail_subsampling

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


def _get_worker_data(data_dir: str) -> KaldiData:
    if data_dir not in _worker_data_cache:
        _worker_data_cache[data_dir] = KaldiData(data_dir)
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

    Y, T, speaker_ids = get_labeledSTFT(
        data, rec, st, ed, frame_size, frame_shift, n_speakers)

    if Y.shape[0] == 0:
        Y_t = np.zeros((0, feature_dim), dtype=np.float32)
    else:
        # specaugment is a training-time random augmentation - never
        # bake it into the precomputed cache, it must stay dynamic.
        Y_t = transform(
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
    parser.add_argument('--chunk-size', type=int, default=2000,
                         help="Raw-frame span of each chunk (NOT "
                              "subsampled domain). If you plan to use "
                              "subsampling=S and want N output frames per "
                              "chunk, pass chunk-size = N * S.")
    parser.add_argument('--context-size', type=int, default=0,
                         help="(load-time) unused here, splice() is cheap "
                              "and applied when loading")
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
                         help="(load-time) unused here -- chunk_size is "
                              "already in raw-frame domain, so subsampling "
                              "is applied freely at load time in "
                              "PrecomputedKaldiDiarizationDataset.")
    parser.add_argument('--use-last-samples', action='store_true')
    parser.add_argument('--min-length', type=int, default=0)
    parser.add_argument('--specaugment', action='store_true',
                         help="(load-time) ignored here; specaugment is "
                              "never precomputed, always applied fresh "
                              "at load time")
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--tail-subsampling', type=int, default=None,
                         help="ONLY for exact-match verification against "
                              "KaldiDiarizationDataset(subsampling=S). "
                              "Reproduces its tail-chunk rounding quirk "
                              "for that specific S. Leave unset for "
                              "normal precompute (keeps full recording, "
                              "no truncation, stays subsampling-agnostic).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s %(levelname)s %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    data = KaldiData(args.data_dir)
    chunk_indices = build_chunk_indices(
        data, args.chunk_size, args.sampling_rate, args.frame_shift,
        args.use_last_samples, args.min_length,
        tail_subsampling=args.tail_subsampling)

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
        'tail_subsampling': args.tail_subsampling,
        'chunk_indices': chunk_indices,
        # chunk_size above is raw-frame domain; subsampling is applied
        # freely at load time and is NOT recorded/enforced here.
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
