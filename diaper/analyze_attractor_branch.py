#!/usr/bin/env python3
"""Analysis of the attractor-existence branch (per_prcvblock/
per_frameenclayer attractors_logits -> counter head) on a dev set, using the
exact same forward pass / PIT-alignment train.py's dev loop uses (reuses
get_loss so exists_mask is the true PIT-aligned target).

Run the same way as eval_checkpoint.py, from the repo root:
    python diaper/analyze_attractor_branch.py -c <config.yaml> \\
        --init-epochs 99-108

<config.yaml> is any train.py-style config with valid_precomputed_dir/
valid_data_dir/valid_features_dir set, plus --init-model-path/--init-epochs
pointing at the checkpoint(s) to evaluate. See eval_checkpoint.py's
docstring for the same argument conventions.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils.collections_abc_compat  # noqa: E402,F401

from backend.losses import get_loss, pad_labels_zeros, pad_sequence  # noqa: E402
from backend.models import average_checkpoints, get_model  # noqa: E402
from eval_checkpoint import get_dev_dataloader  # noqa: E402
from train import parse_arguments  # noqa: E402

import numpy as np  # noqa: E402
import random  # noqa: E402
import threadpoolctl  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

ANALYSIS_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_output")

if __name__ == '__main__':
    args = parse_arguments()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        threadpoolctl.threadpool_limits(limits=args.num_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = torch.device("cuda") if args.gpu >= 1 else torch.device("cpu")

    model = get_model(args)
    # allow_partial=True: tolerates checkpoints saved before spk_counting_head
    # existed -- this script never reads that head's output.
    model = average_checkpoints(
        args.device, model, args.init_model_path, args.init_epochs,
        allow_partial=True)
    model = model.to(args.device)
    model.eval()

    dev_loader = get_dev_dataloader(args)

    all_probs, all_exists, all_nspk, all_names = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dev_loader, total=len(dev_loader)):
            features = batch['xs']
            labels = batch['ts']
            spkids = batch['spk_ids']
            names = batch['names']
            n_speakers = np.asarray([
                max(torch.where(t.sum(0) != 0)[0]) + 1
                if t.sum() > 0 else 0 for t in labels])
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels_zeros(labels, args.n_attractors)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)

            (
                all_frame_embs,
                per_frameenclayer_ys_logits,
                per_frameenclayer_attractors_logits,
                per_frameenclayer_attractors,
                per_prcvblock_ys_logits,
                per_prcvblock_attractors_logits,
                per_prcvblock_attractors,
                per_prcvblock_l2a_entropy_term,
                per_prcvblock_latents
            ) = model.forward(features, args)

            (
                activation_loss_BCE, activation_loss_DER,
                attractor_existence_loss, att_qty_loss,
                vad_loss_v, osd_loss_v, spkid_loss_v, exists_mask
            ) = get_loss(
                per_frameenclayer_ys_logits[:, :, :, -1],
                labels, n_speakers,
                per_frameenclayer_attractors_logits[:, :, -1],
                model,
                per_frameenclayer_attractors[:, :, :, -1],
                args.speakerid_num_speakers, spkids, args)

            if not torch.isfinite(attractor_existence_loss):
                continue

            existence_probs = torch.sigmoid(
                per_frameenclayer_attractors_logits[:, :, -1])
            all_probs.append(existence_probs.cpu().numpy())
            all_exists.append(exists_mask.cpu().numpy())
            all_nspk.append(n_speakers)
            all_names.extend(names)

    probs = np.concatenate(all_probs, axis=0)    # (N, n_attractors)
    exists = np.concatenate(all_exists, axis=0)  # (N, n_attractors)
    nspk = np.concatenate(all_nspk, axis=0)      # (N,)
    names = np.array(all_names)

    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    out_npz = os.path.join(ANALYSIS_OUTPUT_DIR, "attractor_branch.npz")
    np.savez(out_npz, probs=probs, exists=exists, nspk=nspk, names=names)
    print(f"\nSaved {probs.shape[0]} chunks x {probs.shape[1]} attractor "
          f"slots to {out_npz}")

    thr = args.estimate_spk_qty_thr
    pred = (probs > thr).astype(np.float32)
    exists_f = exists.astype(np.float32)

    tp = ((pred == 1) & (exists_f == 1)).sum()
    fp = ((pred == 1) & (exists_f == 0)).sum()
    fn = ((pred == 0) & (exists_f == 1)).sum()
    tn = ((pred == 0) & (exists_f == 0)).sum()
    total = pred.size
    acc = (tp + tn) / total * 100

    print("\n=== Attractor-existence branch: confusion matrix "
          f"(threshold={thr}) ===")
    print(f"  total attractor slots evaluated: {total} "
          f"({probs.shape[0]} chunks x {probs.shape[1]} slots)")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  acc={acc:.2f}%")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    print(f"  precision={precision*100:.2f}%  recall={recall*100:.2f}%  "
          f"(recall low => branch under-activates -> miss-heavy DER; "
          f"precision low => over-activates -> FA-heavy DER)")

    print("\n=== Calibration: mean predicted prob by ground truth ===")
    print(f"  mean prob | exists=1 (real speaker slots): "
          f"{probs[exists_f == 1].mean():.4f} "
          f"(std {probs[exists_f == 1].std():.4f}, n={int(exists_f.sum())})")
    print(f"  mean prob | exists=0 (padding/absent slots): "
          f"{probs[exists_f == 0].mean():.4f} "
          f"(std {probs[exists_f == 0].std():.4f}, "
          f"n={int((exists_f == 0).sum())})")

    print("\n=== Calibration curve (10 equal-width prob bins) ===")
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(probs.ravel(), bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    ef = exists_f.ravel()
    pf = probs.ravel()
    for b in range(10):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        emp_rate = ef[mask].mean()
        mean_pred = pf[mask].mean()
        print(f"  pred in [{bins[b]:.1f},{bins[b+1]:.1f}): n={n:6d}  "
              f"mean_pred={mean_pred:.3f}  empirical_exists_rate={emp_rate:.3f}")

    print("\n=== Per-attractor-slot-index breakdown "
          "(slot order after PIT alignment) ===")
    for s in range(probs.shape[1]):
        slot_exists = exists_f[:, s]
        slot_pred = pred[:, s]
        slot_acc = (slot_pred == slot_exists).mean() * 100
        slot_exist_rate = slot_exists.mean() * 100
        slot_mean_prob = probs[:, s].mean()
        print(f"  slot {s}: exists_rate={slot_exist_rate:5.1f}%  "
              f"mean_prob={slot_mean_prob:.3f}  acc={slot_acc:5.1f}%")

    print("\n=== Predicted vs reference speaker count "
          "(sum of active attractors per chunk) ===")
    pred_qty = pred.sum(axis=1)
    err = pred_qty - nspk
    print(f"  mean ref qty={nspk.mean():.3f}  mean pred qty={pred_qty.mean():.3f}")
    print(f"  mean error (pred-ref)={err.mean():+.3f}  "
          f"mean abs error={np.abs(err).mean():.3f}")
    max_n = int(max(nspk.max(), pred_qty.max())) + 1
    print("  ref_spk_qty -> mean_pred_qty (n_chunks):")
    for n in range(max_n):
        m = nspk == n
        if m.sum() == 0:
            continue
        print(f"    ref={n}: mean_pred={pred_qty[m].mean():.3f}  n_chunks={int(m.sum())}")
