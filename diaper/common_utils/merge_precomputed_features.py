#!/usr/bin/env python3

# Merge two or more precompute_features.py output directories into one, so
# PrecomputedKaldiDiarizationDataset (which only accepts a single
# `precomputed_dir`, see precomputed_diarization_dataset.py) can read them
# as a single dataset. Chunk files are renumbered contiguously (0..N-1) into
# output_dir, and their chunk_indices are concatenated in the same order
# into a merged meta.pkl.
#
# Only meta fields that affect the *content*/shape of stored features
# (feature_dim, frame_shift, frame_size, input_transform, sampling_rate,
# n_speakers) are required to match exactly across inputs -- merging sources
# with different values there would silently corrupt training (e.g.
# mismatched feature_dim breaks batch stacking, mismatched n_speakers means
# label columns mean different things). Fields that only affected
# precompute-time chunk-boundary decisions (chunk_size, use_last_samples,
# min_length, tail_subsampling) and data_dir have no effect once
# chunk_indices exist, so they're kept per-source as lists in the merged
# meta purely for provenance.
#
# No project imports needed -- this only moves/copies pickle files, it
# doesn't touch audio or features.

import argparse
import logging
import os
import pickle
import shutil


REQUIRED_MATCHING_FIELDS = [
    'feature_dim', 'frame_shift', 'frame_size', 'input_transform',
    'sampling_rate', 'n_speakers',
]
PER_SOURCE_FIELDS = [
    'data_dir', 'chunk_size', 'use_last_samples', 'min_length',
    'tail_subsampling',
]


def load_meta(precomputed_dir: str) -> dict:
    with open(os.path.join(precomputed_dir, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def place_chunk(
    src_dir: str,
    src_idx: int,
    dst_dir: str,
    dst_idx: int,
    mode: str,
) -> None:
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
        description="Merge two or more precompute_features.py output "
                     "directories into one, so PrecomputedKaldiDiarizationDataset "
                     "can read them as a single dataset.")
    parser.add_argument('input_dirs', nargs='+',
                         help='two or more directories produced by '
                              'precompute_features.py')
    parser.add_argument('output_dir',
                         help='directory to write the merged dataset to')
    parser.add_argument('--mode', choices=['copy', 'symlink', 'hardlink'],
                         default='copy',
                         help="how to place chunk files in output_dir: "
                              "'copy' (default, safest/most portable, uses "
                              "extra disk), 'hardlink' (no extra disk, "
                              "input_dirs must be on the same filesystem as "
                              "output_dir), 'symlink' (no extra disk, but "
                              "breaks if input_dirs are later moved/deleted)")
    args = parser.parse_args()

    if len(args.input_dirs) < 2:
        parser.error("need at least 2 input_dirs to merge")

    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s %(levelname)s %(message)s')

    metas = [load_meta(d) for d in args.input_dirs]

    reference_dir, reference = args.input_dirs[0], metas[0]
    for src_dir, meta in zip(args.input_dirs[1:], metas[1:]):
        for field in REQUIRED_MATCHING_FIELDS:
            if meta.get(field) != reference.get(field):
                raise ValueError(
                    f"Cannot merge: '{field}' differs between "
                    f"{reference_dir} ({reference.get(field)!r}) and "
                    f"{src_dir} ({meta.get(field)!r}). Merging sources with "
                    "different feature shapes/semantics would corrupt "
                    "training.")

    os.makedirs(args.output_dir, exist_ok=True)

    merged_chunk_indices = []
    dst_idx = 0
    for src_dir, meta in zip(args.input_dirs, metas):
        n = len(meta['chunk_indices'])
        logging.info(f"Merging {n} chunks from {src_dir}")
        for src_idx in range(n):
            place_chunk(src_dir, src_idx, args.output_dir, dst_idx, args.mode)
            dst_idx += 1
        merged_chunk_indices.extend(meta['chunk_indices'])

    merged_meta = {field: reference[field] for field in REQUIRED_MATCHING_FIELDS}
    for field in PER_SOURCE_FIELDS:
        merged_meta[field] = [meta.get(field) for meta in metas]
    merged_meta['source_dirs'] = list(args.input_dirs)
    merged_meta['chunk_indices'] = merged_chunk_indices

    with open(os.path.join(args.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(merged_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(
        f"Done. Merged {len(args.input_dirs)} sources into "
        f"{args.output_dir}: {dst_idx} total chunks.")


if __name__ == '__main__':
    main()
