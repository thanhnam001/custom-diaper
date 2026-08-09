#!/usr/bin/env python3
"""Diagnoses *why* the attractor-existence branch under-activates (see
analyze_attractor_branch.py for the aggregate confusion-matrix/calibration
view first). Distinguishes matching-driven attractor-slot collapse from
generic undertraining of the "losing" slots:

1. Hungarian-matched slot -> reference-speaker-column assignment
   histogram (does the model keep handing the "real speaker" match to
   the same 2-3 slots, or is matching itself broad but only some slots'
   existence signal gets recognized?).
2. Pairwise cosine similarity between the 10 slots' post-training
   attractor vectors (averaged over the dev set) -- do the "loser"
   slots cluster together (collapsed/undifferentiated) while a few
   stand apart?
3. Zero-forward-pass inspection of latents2attractors.weights (the
   n_latents x n_attractors softmax-combination matrix that is the
   ONLY per-slot-differentiating parameter upstream of the attractors)
   and of the existence head (counter) to confirm it has no per-slot
   parameters at all.
4. Per-slot mean existence probability sliced by low vs high reference
   speaker count, to see whether the "winner" slots are fixed
   regardless of context.

Run the same way as eval_checkpoint.py, from the repo root:
    python diaper/attractor_collapse_analysis.py -c <config.yaml> \\
        --init-epochs 99-108
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils.collections_abc_compat  # noqa: E402,F401

from backend.losses import pad_labels_zeros, pad_sequence, pit_loss_multispk  # noqa: E402
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

    base = model.module if hasattr(model, 'module') else model

    # ---------------------------------------------------------------
    # 3. Checkpoint-only weight inspection (zero compute over data) --
    # run first since it's cheap and useful even without a dev set.
    # ---------------------------------------------------------------
    n_attractors = args.n_attractors
    print("\n=== 3. latents2attractors.weights (n_latents x n_attractors "
          "softmax-combination matrix) -- the only per-slot-differentiating "
          "parameter upstream of the attractor vectors ===")
    l2a = base.latents2attractors
    if hasattr(l2a, 'weights'):
        w = l2a.weights.detach().cpu()  # (n_latents, n_attractors)
        w_softmax = torch.softmax(w, dim=0).numpy()  # per-column distribution over latents
        n_latents = w_softmax.shape[0]
        print(f"weights shape: {tuple(w.shape)} (n_latents={n_latents}, "
              f"n_attractors={n_attractors})")
        print("per-slot softmax-distribution stats over the latents:")
        for s in range(n_attractors):
            col = w_softmax[:, s]
            entropy = -np.sum(col * np.log(col + 1e-12))
            max_w = col.max()
            eff_n = 1.0 / np.sum(col ** 2)  # inverse participation ratio
            print(f"  slot {s}: entropy={entropy:.3f} (uniform max="
                  f"{np.log(n_latents):.3f})  max_weight={max_w:.4f}  "
                  f"effective_n_latents={eff_n:6.1f}")
    else:
        print(f"latents2attractors is {type(l2a).__name__}, no 'weights' "
              f"parameter (lat2att != weighted_average)")

    print("\nexistence head (self.counter) parameters -- applied identically "
          "to every slot's attractor vector (no per-slot weights exist here "
          "by construction):")
    print(f"  counter.weight shape: {tuple(base.counter.weight.shape)} "
          f"(single (1, d_latents) row, shared across all {n_attractors} slots)")
    print(f"  counter.weight norm: {base.counter.weight.detach().norm().item():.4f}  "
          f"counter.bias: {base.counter.bias.detach().item():.4f}")

    # ---------------------------------------------------------------
    # Part A: dev-set forward pass -- probs, exists, ref_alig (matched
    # reference column index per slot), attractor vectors, nspk.
    # ---------------------------------------------------------------
    dev_loader = get_dev_dataloader(args)

    all_probs, all_exists, all_refalig, all_vecs, all_nspk, all_names = \
        [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dev_loader, total=len(dev_loader), desc="dev pass"):
            features = batch['xs']
            labels = batch['ts']
            names = batch['names']
            n_speakers = np.asarray([
                max(torch.where(t.sum(0) != 0)[0]) + 1
                if t.sum() > 0 else 0 for t in labels])
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels_zeros(labels, n_attractors)
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

            ys_logits = per_frameenclayer_ys_logits[:, :, :, -1]
            att_logits = per_frameenclayer_attractors_logits[:, :, -1]
            att_vecs = per_frameenclayer_attractors[:, :, :, -1]  # (B, n_att, d_lat)

            max_n_speakers = max(n_speakers)
            ts_padded = torch.stack(pad_labels_zeros(labels, max_n_speakers))
            logits_padded = torch.stack(pad_labels_zeros(ys_logits, max_n_speakers))

            (_, _, _, exists_mask, permutations) = pit_loss_multispk(
                logits_padded, ts_padded, att_logits, n_speakers, args)
            if not torch.isfinite(exists_mask).all():
                continue

            existence_probs = torch.sigmoid(att_logits)

            # permutations: (B, n_attractors) -- for slot j, which target
            # column (post outer n_attractors-padding, pre per-batch
            # max_n_speakers-padding) it was Hungarian-matched to.
            all_probs.append(existence_probs.cpu().numpy())
            all_exists.append(exists_mask.cpu().numpy())
            all_refalig.append(permutations.cpu().numpy())
            all_vecs.append(att_vecs.cpu().numpy())
            all_nspk.append(n_speakers)
            all_names.extend(names)

    probs = np.concatenate(all_probs, axis=0)       # (N, n_att)
    exists = np.concatenate(all_exists, axis=0)      # (N, n_att)
    refalig = np.concatenate(all_refalig, axis=0)    # (N, n_att)
    vecs = np.concatenate(all_vecs, axis=0)          # (N, n_att, d_lat)
    nspk = np.concatenate(all_nspk, axis=0)          # (N,)
    names = np.array(all_names)

    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    out_npz = os.path.join(ANALYSIS_OUTPUT_DIR, "attractor_collapse.npz")
    np.savez(
        out_npz, probs=probs, exists=exists, refalig=refalig, vecs=vecs,
        nspk=nspk, names=names)
    print(f"\nSaved {probs.shape[0]} chunks to {out_npz}")

    # ---------------------------------------------------------------
    # 1. Slot -> matched reference-column histogram (real matches only)
    # ---------------------------------------------------------------
    print("\n=== 1. Hungarian-matched slot -> reference-column histogram "
          "(real matches only, exists=1) ===")
    max_refcol = int(refalig.max()) + 1
    hist = np.zeros((n_attractors, max_refcol), dtype=int)
    for i in range(refalig.shape[0]):
        for s in range(n_attractors):
            if exists[i, s] == 1:
                hist[s, refalig[i, s]] += 1
    header = "slot\\refcol " + " ".join(f"{c:5d}" for c in range(max_refcol)) + "   row_total"
    print(header)
    for s in range(n_attractors):
        row_total = hist[s].sum()
        print(f"slot {s:2d}      " +
              " ".join(f"{hist[s, c]:5d}" for c in range(max_refcol)) +
              f"   {row_total:5d}")
    print("(row_total = how many dev chunks matched this slot to SOME real "
          "reference speaker; column = which target-column index it was)")

    # ---------------------------------------------------------------
    # 2. Pairwise cosine similarity between slots' mean attractor vectors
    # ---------------------------------------------------------------
    print("\n=== 2. Pairwise cosine similarity between slots' attractor "
          "vectors ===")
    norms = np.linalg.norm(vecs, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vecs = vecs / norms
    mean_dir = unit_vecs.mean(axis=0)  # (n_att, d_lat) -- average direction per slot
    mean_dir_norm = mean_dir / np.linalg.norm(mean_dir, axis=1, keepdims=True)
    cos_sim = mean_dir_norm @ mean_dir_norm.T
    print("mean-direction cosine similarity matrix (10x10):")
    print("      " + " ".join(f"s{c:<5d}" for c in range(n_attractors)))
    for s in range(n_attractors):
        print(f"s{s:<3d} " + " ".join(f"{cos_sim[s, c]:6.3f}" for c in range(n_attractors)))

    print("\nwithin-slot cross-chunk consistency (mean cosine sim of the "
          "slot's vector across random chunk pairs) and raw vector norm:")
    rng = np.random.default_rng(0)
    n_pairs = min(2000, unit_vecs.shape[0] * (unit_vecs.shape[0] - 1) // 2)
    idx_a = rng.integers(0, unit_vecs.shape[0], n_pairs)
    idx_b = rng.integers(0, unit_vecs.shape[0], n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]
    for s in range(n_attractors):
        sims = np.sum(unit_vecs[idx_a, s, :] * unit_vecs[idx_b, s, :], axis=1)
        mean_prob_slot = probs[:, s].mean()
        print(f"  slot {s}: within-slot cos_sim={sims.mean():.3f} "
              f"(std {sims.std():.3f})  mean_raw_norm={norms[:, s, 0].mean():.3f}  "
              f"mean_exist_prob={mean_prob_slot:.3f}")

    # ---------------------------------------------------------------
    # 4. Per-slot mean existence prob, low vs high reference speaker count
    # ---------------------------------------------------------------
    print("\n=== 4. Per-slot mean existence probability: low vs high "
          "reference speaker count ===")
    median_nspk = np.median(nspk)
    low_mask = nspk <= 3
    high_mask = nspk >= 7
    print(f"  low group (nspk<=3): n={int(low_mask.sum())}  "
          f"high group (nspk>=7): n={int(high_mask.sum())}  "
          f"(median nspk in dev set = {median_nspk})")
    low_means = probs[low_mask].mean(axis=0) if low_mask.sum() > 0 else np.full(n_attractors, np.nan)
    high_means = probs[high_mask].mean(axis=0) if high_mask.sum() > 0 else np.full(n_attractors, np.nan)
    low_rank = np.argsort(-low_means)
    high_rank = np.argsort(-high_means)
    print("  slot: mean_prob(low nspk)  mean_prob(high nspk)")
    for s in range(n_attractors):
        print(f"  slot {s}: {low_means[s]:.3f}                {high_means[s]:.3f}")
    print(f"  top-3 winner slots | low nspk group:  {list(low_rank[:3])}")
    print(f"  top-3 winner slots | high nspk group: {list(high_rank[:3])}")
