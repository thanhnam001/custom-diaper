#!/usr/bin/env python3
"""Answers: if the number of speakers per file were known and attractors
were selected by oracle top-K quantity (instead of the deployed
existence-probability threshold), how much would DER drop -- and separately,
if attractor SELECTION were 100% correct (not just the count, but exactly
which slots correspond to which real speaker, via the same Hungarian
matching pit_loss_multispk/get_exists_mask use), what DER floor remains.

Reuses the exact model/checkpoint/postprocessing path infer.py uses to
produce RTTMs, but instead of writing RTTMs it computes frame-level DER
(calculate_metrics, same as infer.py --compute-metrics) for three attractor
active-mask variants per file, holding everything else (per-frame threshold,
median filter, y_probs themselves) identical:

  threshold   -- deployed behavior: existence_probs >= estimate_spk_qty_thr
  oracle_topk -- top-K existence_probs, K = true reference speaker count
                 for that file (answers "know the count, pick by quantity")
  oracle_perfect -- the Hungarian-matched ground-truth active mask itself
                 (get_exists_mask): exactly the right slots, by construction
                 (answers "attractor branch is 100% correct")

Run from the repo root:
    python diaper/oracle_attractor_analysis.py \\
        -c models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31/infer_msdwild_mlp_unmaskeddiv.yaml \\
        --infer-data-dir <local msdwild test kaldi dir> \\
        --models-path <local checkpoint dir>
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils.collections_abc_compat  # noqa: E402,F401

from backend.losses import pad_labels_zeros  # noqa: E402
from backend.models import average_checkpoints, get_model  # noqa: E402
from common_utils.metrics import calculate_metrics  # noqa: E402
from infer import (  # noqa: E402
    estimate_diarization_outputs, get_exists_mask, get_hard_decisions,
    get_infer_dataloader, parse_arguments,
)

import numpy as np  # noqa: E402
import random  # noqa: E402
import threadpoolctl  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

VARIANTS = ["threshold", "oracle_topk", "oracle_perfect"]
METRIC_KEYS = ["DER", "DER_miss", "DER_FA", "DER_conf"]


def build_active_mask(variant, existence_probs, y_probs, ref_padded, n_frames, args):
    if variant == "threshold":
        mask = torch.zeros_like(existence_probs)
        mask[existence_probs >= args.estimate_spk_qty_thr] = 1.0
        return mask
    active_cols = torch.where(ref_padded[:n_frames].sum(axis=0) != 0)[0]
    k_ref = int(active_cols.max().item()) + 1 if active_cols.numel() > 0 else 0
    if variant == "oracle_topk":
        mask = torch.zeros_like(existence_probs)
        if k_ref > 0:
            _, order = torch.sort(existence_probs, descending=True)
            mask[order[:k_ref]] = 1.0
        return mask
    if variant == "oracle_perfect":
        return get_exists_mask(
            y_probs[:n_frames].to("cpu"), ref_padded[:n_frames].to("cpu")
        ).to(existence_probs.device)
    raise ValueError(variant)


if __name__ == '__main__':
    args = parse_arguments()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        threadpoolctl.threadpool_limits(limits=args.num_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = torch.device("cuda") if args.gpu >= 1 else torch.device("cpu")
    # estimate_diarization_outputs() only asserts one of these is set; the
    # actual selection rule used for the "threshold" variant below is
    # re-derived from args.estimate_spk_qty_thr directly, not from its
    # ys_active return value (which we ignore).
    if args.estimate_spk_qty_thr == -1 and args.estimate_spk_qty == -1:
        args.estimate_spk_qty_thr = 0.5

    infer_loader = get_infer_dataloader(args)

    model = get_model(args)
    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs, allow_partial=True)
    model = model.to(args.device)
    model.eval()

    collars = [0.0, 0.25]
    acc = {c: {v: {k: 0.0 for k in METRIC_KEYS} for v in VARIANTS} for c in collars}
    n_scored = 0

    frame_period = args.subsampling * args.frame_shift / args.sampling_rate

    with torch.no_grad():
        for batch in tqdm(infer_loader, total=len(infer_loader)):
            name = batch['names'][0]
            input = torch.stack(batch['xs']).to(args.device)
            try:
                (_, existence_probs, _, _, y_probs) = estimate_diarization_outputs(
                    model, input, args)
            except RuntimeError as e:
                print(f"{name}: forward failed, skipping ({e})")
                continue
            existence_probs = existence_probs[0]
            y_probs = y_probs[0]
            ref_labels = batch['ts'][0]
            ref_padded = pad_labels_zeros([ref_labels], args.n_attractors)[0]
            n_frames = min(y_probs.shape[0], ref_padded.shape[0])
            if n_frames == 0:
                continue

            for variant in VARIANTS:
                mask = build_active_mask(
                    variant, existence_probs, y_probs, ref_padded, n_frames, args)
                pred_hard = get_hard_decisions(
                    y_probs[:n_frames] * mask.unsqueeze(0),
                    args.threshold, args.median_window_length,
                    args.normalize_probs)
                pred_hard = torch.from_numpy(pred_hard).float()
                for collar in collars:
                    collar_frames = round(collar / frame_period) if collar > 0 else 0
                    file_metrics, denom = calculate_metrics(
                        ref_padded[:n_frames].unsqueeze(0),
                        pred_hard.unsqueeze(0),
                        threshold=0.5, collar_frames=collar_frames,
                        return_denominators=True)
                    if denom["speech_frames_tot"] == 0:
                        continue
                    for k in METRIC_KEYS:
                        acc[collar][variant][k] += float(file_metrics[k])
            n_scored += 1

    print(f"\nn_files_scored={n_scored}\n")
    for collar in collars:
        print(f"--- collar={collar}s ---")
        for variant in VARIANTS:
            avg = {k: acc[collar][variant][k] / n_scored for k in METRIC_KEYS}
            print(
                f"{variant:>15}: DER={avg['DER']:.2f}%  "
                f"(miss={avg['DER_miss']:.2f} fa={avg['DER_FA']:.2f} "
                f"conf={avg['DER_conf']:.2f})")
        print()
