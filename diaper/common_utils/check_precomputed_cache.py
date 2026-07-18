#!/usr/bin/env python3

# Integrity checker for a precompute_features.py output directory.
#
# Motivation: KaldiData.load_wav prints "<path> not found" and returns EMPTY
# audio instead of raising when a wav path in wav.scp does not resolve (e.g.
# wrong relative paths after uploading a data dir to Kaggle). precompute
# then silently writes an EMPTY chunk pkl, and at train time
# PrecomputedKaldiDiarizationDataset silently substitutes the previous
# chunk (self.saved) for every empty one. A cache poisoned this way trains
# without any error message -- the model just never learns. This script
# makes that failure mode (and a few others) visible.
#
# Usage:
#   python check_precomputed_cache.py <precomputed_dir> [<precomputed_dir2> ...]
#
# Standalone: only needs numpy + pickle (safe to copy to Kaggle next to
# precompute_features.py).

import os
import pickle
import sys

import numpy as np

try:
    # lets numpy<2 read caches written under numpy>=2; no-op otherwise
    import numpy2_pickle_compat  # noqa: F401
except ImportError:
    try:
        import diaper.common_utils.numpy2_pickle_compat  # noqa: F401
    except ImportError:
        pass


def check(d: str) -> bool:
    print(f"\n=== {d} ===")
    meta_path = os.path.join(d, 'meta.pkl')
    if not os.path.isfile(meta_path):
        print("FAIL: no meta.pkl")
        return False
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    ci = meta['chunk_indices']
    print(f"meta: n_speakers={meta.get('n_speakers')} "
          f"chunk_size={meta.get('chunk_size')} "
          f"feature_dim={meta.get('feature_dim')} "
          f"input_transform={meta.get('input_transform')} "
          f"tail_subsampling={meta.get('tail_subsampling')} "
          f"#chunks={len(ci)}")

    pkls = [f for f in os.listdir(d)
            if f.endswith('.pkl') and f != 'meta.pkl']
    if len(pkls) != len(ci):
        print(f"FAIL: meta lists {len(ci)} chunks but directory has "
              f"{len(pkls)} chunk pkls")
        return False

    n_empty, n_nonfinite, n_zero_labels, n_index_mismatch = 0, 0, 0, 0
    empty_examples, frames, act_fracs = [], [], []
    y_means, y_stds = [], []
    for i in range(len(ci)):
        path = os.path.join(d, f"{i:08d}.pkl")
        if not os.path.isfile(path):
            print(f"FAIL: missing chunk file {path}")
            return False
        with open(path, 'rb') as f:
            c = pickle.load(f)
        Y, T = c['Y'], c['T']
        rec, st, ed = ci[i]
        if (c['rec'], c['st'], c['ed']) != (rec, st, ed):
            n_index_mismatch += 1
        frames.append(Y.shape[0])
        if Y.shape[0] == 0:
            n_empty += 1
            if len(empty_examples) < 5:
                empty_examples.append((i, rec))
            continue
        if not np.isfinite(Y).all():
            n_nonfinite += 1
        if T.sum() == 0:
            n_zero_labels += 1
        act_fracs.append(float((T.sum(axis=1) > 0).mean()))
        y_means.append(float(Y.mean()))
        y_stds.append(float(Y.std()))

    frames = np.asarray(frames)
    ok = True
    if n_empty:
        ok = False
        print(f"FAIL: {n_empty}/{len(ci)} chunks are EMPTY (audio was not "
              f"readable at precompute time). Examples: {empty_examples}")
        print("      -> re-check wav.scp paths in the source data dir and "
              "re-run precompute; grep its log for 'not found'.")
    else:
        print("OK: no empty chunks")
    if n_nonfinite:
        ok = False
        print(f"FAIL: {n_nonfinite} chunks contain NaN/Inf features")
    else:
        print("OK: all features finite")
    if n_index_mismatch:
        ok = False
        print(f"FAIL: {n_index_mismatch} chunks whose stored rec/st/ed "
              f"disagree with meta chunk_indices (corrupted/mixed cache?)")
    else:
        print("OK: chunk files consistent with meta chunk_indices")
    print(f"labels: {n_zero_labels} chunks with all-zero labels "
          f"({100 * n_zero_labels / max(len(ci), 1):.1f}% -- a few is normal, "
          f"many means broken segments/utt2spk)")
    if len(act_fracs):
        act = np.asarray(act_fracs)
        print(f"speech-activity fraction per chunk: "
              f"min={act.min():.3f} mean={act.mean():.3f} max={act.max():.3f}")
        print(f"feature stats across chunks: mean(Y.mean)={np.mean(y_means):.4f} "
              f"mean(Y.std)={np.mean(y_stds):.4f}")
    full = int((frames == meta.get('chunk_size', -1)).sum())
    print(f"frames per chunk: min={frames.min()} max={frames.max()} "
          f"(#full={full}, #other={len(ci) - full})")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__ or "usage: check_precomputed_cache.py <dir> [...]")
    all_ok = all([check(d) for d in sys.argv[1:]])
    sys.exit(0 if all_ok else 1)
