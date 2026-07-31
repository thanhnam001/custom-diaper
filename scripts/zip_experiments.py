#!/usr/bin/env python3
# Zip experiment folders into a single archive for transfer, keeping only
# the N most recent checkpoints per checkpoint dir (tensorboard logs /
# configs / everything else is zipped unchanged). Originals are never
# modified.
#
# Usage:
#   python scripts/zip_experiments.py ROOT [ROOT ...] -o OUT.zip [--keep-n 10]
#
# Every directory literally named `models` anywhere under a ROOT (at any
# depth) is treated as an independent checkpoint dir and pruned on its own
# -- this covers not just a plain pretrain/adapt exp's own <output_path>/
# models/, but also sibling finetune stages whose output_path points *inside*
# that same exp dir, e.g. <adapt_exp>/models_finetuneRAMC/models/ and
# <adapt_exp>/models_finetuneMSDWILD/models/ sitting next to the adapt
# stage's own <adapt_exp>/models/. Checkpoints (checkpoint_*.tar) are ranked
# by the epoch number embedded in the filename (models.py's
# save_checkpoint() writes `checkpoint_{epoch}.tar`, epoch a float) rather
# than mtime, since mtime doesn't survive a copy/rsync intact; only the
# --keep-n highest-epoch checkpoints per models/ dir are included.
#
# Everything found across all ROOTs goes into one archive, each file stored
# under a path relative to its ROOT's parent (so same-named dirs under
# different ROOTs don't collide).

import argparse
import os
import re
import zipfile
from pathlib import Path

from tqdm import tqdm

CHECKPOINT_RE = re.compile(r'^checkpoint_(-?\d+(?:\.\d+)?)\.tar$')


def checkpoint_epoch(path: Path):
    m = CHECKPOINT_RE.match(path.name)
    return float(m.group(1)) if m else None


def find_models_dirs(root: Path):
    """Every dir literally named `models` at or under root, regardless of
    nesting depth (unlike os.walk pruning, does NOT stop descending once
    one is found -- a finetune stage's models/ can sit inside its parent
    exp dir, alongside that exp's own models/)."""
    dirs = [p for p in root.rglob('models') if p.is_dir()]
    if root.name == 'models':
        dirs.append(root)
    return dirs


def checkpoints_to_skip(models_dir: Path, keep_n: int) -> set:
    all_ckpts = list(models_dir.glob('checkpoint_*.tar'))
    dated = []
    unparseable = []
    for p in all_ckpts:
        epoch = checkpoint_epoch(p)
        if epoch is None:
            unparseable.append(p)
        else:
            dated.append((epoch, p))

    if unparseable:
        print(f"  {models_dir}: {len(unparseable)} checkpoint file(s) "
              "don't match `checkpoint_<epoch>.tar` -- keeping them "
              f"unconditionally: {[p.name for p in unparseable]}")

    skip = set()
    if keep_n > 0 and len(dated) > keep_n:
        dated.sort(key=lambda t: t[0])
        skip = {p for _, p in dated[:-keep_n]}

    print(f"  {models_dir}: {len(all_ckpts)} checkpoint(s) found, "
          f"keeping {len(all_ckpts) - len(skip)}, skipping {len(skip)}")
    return skip


def collect_files(roots, keep_n: int):
    """Returns (file_list, n_checkpoints_skipped, n_symlinks_skipped) where
    file_list is [(abs_path, arcname), ...] for everything to go in the
    zip. arcname is abs_path relative to its ROOT's parent."""
    file_list = []
    n_checkpoints_skipped = 0
    n_symlinks_skipped = 0

    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            print(f"skipping {root}: not a directory")
            continue

        models_dirs = find_models_dirs(root)
        if not models_dirs:
            print(f"note: no `models` dir found anywhere under {root} "
                  "-- nothing will be pruned, everything gets zipped as-is")

        skip = set()
        for models_dir in models_dirs:
            this_skip = checkpoints_to_skip(models_dir, keep_n)
            n_checkpoints_skipped += len(this_skip)
            skip |= this_skip

        for p in root.rglob('*'):
            if p in skip:
                continue
            if p.is_symlink():
                # Not dereferenced -- matches `du`'s default (no
                # --dereference) and avoids silently pulling a symlink's
                # target (which may live entirely outside root, e.g. a
                # shared init checkpoint) into the archive.
                n_symlinks_skipped += 1
                print(f"    skipping symlink {p} -> {os.readlink(p)}")
            elif p.is_file():
                file_list.append((p, p.relative_to(root.parent)))

    return file_list, n_checkpoints_skipped, n_symlinks_skipped


def write_zip(file_list, out_path: Path, compress_level):
    if compress_level is None:
        zip_kwargs = dict(compression=zipfile.ZIP_STORED)
    else:
        zip_kwargs = dict(
            compression=zipfile.ZIP_DEFLATED, compresslevel=compress_level)

    total_bytes = sum(p.stat().st_size for p, _ in file_list)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', **zip_kwargs) as zf, \
            tqdm(total=total_bytes, unit='B', unit_scale=True,
                 desc=out_path.name) as pbar:
        for p, arcname in file_list:
            zf.write(p, arcname)
            pbar.update(p.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exp_dirs', nargs='+', type=Path,
                         help='experiment folders, or roots to search '
                         'recursively for `models` checkpoint dirs')
    parser.add_argument('-o', '--out', type=Path, required=True,
                         help='output .zip path')
    parser.add_argument('--keep-n', type=int, default=10,
                         help='most recent checkpoints to keep per models/ '
                         'dir (default: 10); <= 0 keeps every checkpoint')
    parser.add_argument('--compress-level', type=int, default=None,
                         choices=range(0, 10), metavar='0-9',
                         help='enable DEFLATE compression at this level '
                         '(default: off, i.e. ZIP_STORED). Checkpoint '
                         'tensors barely compress, so the default just '
                         'stores files uncompressed for speed; only set '
                         'this if the archive is mostly compressible '
                         'non-checkpoint data and you want a smaller file.')
    args = parser.parse_args()

    file_list, n_ckpt_skipped, n_symlinks_skipped = collect_files(
        args.exp_dirs, args.keep_n)
    if not file_list:
        print("nothing to zip")
        return

    write_zip(file_list, args.out, args.compress_level)
    print(f"wrote {args.out} ({len(file_list)} files, "
          f"skipped {n_ckpt_skipped} old checkpoints, "
          f"{n_symlinks_skipped} symlinks)")


if __name__ == '__main__':
    main()
