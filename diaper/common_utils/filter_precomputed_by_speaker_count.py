#!/usr/bin/env python3

# Filter a precompute_features.py output directory down to chunks with at
# least --min-speakers distinct speakers, so a large re-chunked source
# (e.g. the 300h simulated 1-10 speaker adapt-phase data, re-precomputed
# at a smaller chunk-size to be mergeable with a real finetuning corpus --
# see merge_precomputed_features.py) can be trimmed to just the
# high-speaker-count chunks it was pulled in for, instead of diluting the
# real data with a large volume of low-speaker-count chunks the real
# corpus already covers on its own.
#
# "Distinct speakers in a chunk" is computed the same way
# analyze_attractor_branch.py / this project's ad-hoc speaker-count checks
# do: a T column counts as present if it's ever 1 anywhere in the chunk
# (T.sum(axis=0) > 0).sum() -- NOT simultaneous-overlap count.
#
# Output directory format matches precompute_features.py exactly (numbered
# 00000000.pkl.. chunk files + meta.pkl with chunk_indices), so it can be
# used directly as --train-precomputed-dir, or as an input to
# merge_precomputed_features.py alongside the real corpus.
#
# Usage:
#   python diaper/common_utils/filter_precomputed_by_speaker_count.py \
#       <input_dir> <output_dir> --min-speakers 5

import argparse
import logging
import os
import pickle
import shutil

import numpy as np


def place_chunk(src_dir: str, src_idx: int, dst_dir: str, dst_idx: int, mode: str) -> None:
    src_path = os.path.join(src_dir, f"{src_idx:08d}.pkl")
    dst_path = os.path.join(dst_dir, f"{dst_idx:08d}.pkl")
    if mode == 'copy':
        shutil.copyfile(src_path, dst_path)
    elif mode == 'hardlink':
        os.link(src_path, dst_path)
    elif mode == 'symlink':
        os.symlink(os.path.abspath(src_path), dst_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter a precompute_features.py output directory down "
                     "to chunks with at least --min-speakers distinct "
                     "speakers.")
    parser.add_argument('input_dir',
                         help='directory produced by precompute_features.py')
    parser.add_argument('output_dir',
                         help='directory to write the filtered dataset to')
    parser.add_argument('--min-speakers', type=int, default=5,
                         help='keep only chunks with at least this many '
                              'distinct speakers (default: 5)')
    parser.add_argument('--mode', choices=['copy', 'symlink', 'hardlink'],
                         default='copy',
                         help="how to place kept chunk files in output_dir: "
                              "'copy' (default, safest/most portable, uses "
                              "extra disk), 'hardlink' (no extra disk, "
                              "input_dir must be on the same filesystem as "
                              "output_dir), 'symlink' (no extra disk, but "
                              "breaks if input_dir is later moved/deleted)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s %(levelname)s %(message)s')

    meta_path = os.path.join(args.input_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    chunk_indices = meta['chunk_indices']
    n_total = len(chunk_indices)

    os.makedirs(args.output_dir, exist_ok=True)

    kept_src_indices = []
    kept_chunk_indices = []
    hist = {}  # distinct-speaker-count -> n_chunks, over ALL chunks (kept or not)
    n_empty = 0

    for src_idx in range(n_total):
        src_path = os.path.join(args.input_dir, f"{src_idx:08d}.pkl")
        with open(src_path, 'rb') as f:
            d = pickle.load(f)
        t = np.asarray(d['T'])
        if t.size == 0:
            n_empty += 1
            continue
        n_speakers_here = int((t.sum(axis=0) > 0).sum())
        hist[n_speakers_here] = hist.get(n_speakers_here, 0) + 1
        if n_speakers_here >= args.min_speakers:
            kept_src_indices.append(src_idx)
            kept_chunk_indices.append(chunk_indices[src_idx])

        if (src_idx + 1) % 1000 == 0:
            logging.info(f"scanned {src_idx + 1}/{n_total} chunks, "
                         f"{len(kept_src_indices)} kept so far")

    logging.info(f"scanned all {n_total} chunks ({n_empty} empty, skipped)")
    logging.info("distinct-speakers-per-chunk histogram (all scanned chunks): "
                 f"{dict(sorted(hist.items()))}")
    logging.info(f"keeping {len(kept_src_indices)}/{n_total} chunks "
                 f"(>= {args.min_speakers} distinct speakers)")

    for dst_idx, src_idx in enumerate(kept_src_indices):
        place_chunk(args.input_dir, src_idx, args.output_dir, dst_idx, args.mode)

    out_meta = {k: v for k, v in meta.items() if k != 'chunk_indices'}
    out_meta['chunk_indices'] = kept_chunk_indices
    out_meta['filtered_from'] = args.input_dir
    out_meta['filtered_min_speakers'] = args.min_speakers
    out_meta['filtered_source_n_chunks'] = n_total

    with open(os.path.join(args.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(out_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Done. Wrote {len(kept_src_indices)} chunks to {args.output_dir}")


if __name__ == '__main__':
    main()
