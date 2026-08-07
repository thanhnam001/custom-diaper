#!/usr/bin/env python3

# Sanity-check that PrecomputedKaldiDiarizationDataset reproduces the
# same output as KaldiDiarizationDataset.
#
# IMPORTANT: chunk_size means different things in the two datasets:
#   - KaldiDiarizationDataset: chunk_size is in the SUBSAMPLED domain,
#     raw span = chunk_size * subsampling
#   - PrecomputedKaldiDiarizationDataset (via precompute_features.py):
#     chunk_size passed to precompute is already the RAW span.
# So to compare apples-to-apples, set:
#     original_chunk_size = precompute_chunk_size / subsampling
# e.g. precompute --chunk-size 6000, subsampling=10
#      -> original KaldiDiarizationDataset(chunk_size=600, ...)
#
# Also set specaugment=False on BOTH sides for this check, since it's a
# random augmentation and will never match between two independent calls.

import argparse
import numpy as np
from tqdm import tqdm
import torch

from diaper.common_utils.diarization_dataset import KaldiDiarizationDataset
from diaper.common_utils.precomputed_diarization_dataset import PrecomputedKaldiDiarizationDataset


def compare_item(a, b, idx, atol=1e-5, rtol=1e-5):
    Y_a, T_a, rec_a, st_a, ed_a, spk_a = a
    Y_b, T_b, rec_b, st_b, ed_b, spk_b = b

    problems = []

    if rec_a != rec_b:
        problems.append(f"rec mismatch: {rec_a} vs {rec_b}")
    if st_a != st_b:
        problems.append(f"st mismatch: {st_a} vs {st_b}")
    if ed_a != ed_b:
        problems.append(f"ed mismatch: {ed_a} vs {ed_b}")
    if list(spk_a) != list(spk_b):
        problems.append(f"speaker_ids mismatch: {spk_a} vs {spk_b}")

    if Y_a.shape != Y_b.shape:
        problems.append(f"Y shape mismatch: {tuple(Y_a.shape)} vs {tuple(Y_b.shape)}")
    else:
        if not torch.allclose(Y_a, Y_b, atol=atol, rtol=rtol):
            max_diff = (Y_a - Y_b).abs().max().item()
            problems.append(f"Y values differ, max abs diff = {max_diff}")

    if T_a.shape != T_b.shape:
        problems.append(f"T shape mismatch: {tuple(T_a.shape)} vs {tuple(T_b.shape)}")
    else:
        if not torch.equal(T_a, T_b):
            n_diff = (T_a != T_b).sum().item()
            problems.append(f"T values differ in {n_diff} entries")

    if problems:
        print(f"[MISMATCH] idx={idx}")
        for p in problems:
            print(f"    - {p}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare KaldiDiarizationDataset vs "
                     "PrecomputedKaldiDiarizationDataset item-by-item.")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('precomputed_dir', type=str)

    # Original-dataset config (chunk_size in SUBSAMPLED domain, as usual)
    parser.add_argument('--chunk-size', type=int, required=True)
    parser.add_argument('--context-size', type=int, default=0)
    parser.add_argument('--feature-dim', type=int, required=True)
    parser.add_argument('--frame-shift', type=int, required=True)
    parser.add_argument('--frame-size', type=int, required=True)
    parser.add_argument('--input-transform', type=str, required=True)
    parser.add_argument('--n-speakers', type=int, default=None)
    parser.add_argument('--sampling-rate', type=int, required=True)
    parser.add_argument('--subsampling', type=int, default=1)
    parser.add_argument('--use-last-samples', action='store_true')
    parser.add_argument('--min-length', type=int, default=0)

    parser.add_argument('--num-items', type=int, default=None,
                         help="Only check the first N items (default: all)")
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    args = parser.parse_args()

    original_ds = KaldiDiarizationDataset(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.n_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=False,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
        specaugment=False,
    )

    precomputed_ds = PrecomputedKaldiDiarizationDataset(
        precomputed_dir=args.precomputed_dir,
        context_size=args.context_size,
        n_speakers=args.n_speakers,
        subsampling=args.subsampling,
        specaugment=False,
    )

    if len(original_ds) != len(precomputed_ds):
        print(f"[WARNING] length mismatch: original={len(original_ds)} "
              f"precomputed={len(precomputed_ds)}")

    n = len(original_ds)
    if args.num_items is not None:
        n = min(n, args.num_items)

    n_ok = 0
    n_bad = 0
    for i in tqdm(range(n)):
        a = original_ds[i]
        b = precomputed_ds[i]
        if compare_item(a, b, i, atol=args.atol, rtol=args.rtol):
            n_ok += 1
        else:
            n_bad += 1

    print(f"\nChecked {n} items: {n_ok} OK, {n_bad} MISMATCH")
    if n_bad == 0:
        print("All good - precomputed dataset matches the original.")
    else:
        raise SystemExit(1)


if __name__ == '__main__':
    main()