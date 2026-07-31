#!/usr/bin/env python3
# Zip experiment folders into a single archive for transfer, keeping only
# the N most recent checkpoints per experiment (tensorboard logs / configs /
# everything else under each exp folder is zipped unchanged). Originals are
# never modified.
#
# Usage:
#   python scripts/zip_experiments.py ROOT [ROOT ...] -o OUT.zip [--keep-n 10]
#
# Each ROOT is searched recursively for experiment directories (any folder
# containing a models/ subdir, e.g. <output_path> from train.yaml -- ROOT
# itself counts if it directly has one). All experiments found across all
# ROOTs go into one archive, each stored under a path relative to that
# ROOT's parent (so same-named experiments in different families don't
# collide). Checkpoints under <exp_dir>/models/checkpoint_*.tar are ranked
# by mtime (mirrors the sort train.py itself uses to find the latest
# checkpoint, see train.py's `paths.sort(key=lambda x: os.path.getmtime(x))`);
# only the --keep-n newest are included per experiment.

import argparse
import os
import zipfile
from pathlib import Path

from tqdm import tqdm


def discover_experiment_dirs(root: Path):
    """Find every dir under (and including) root that has a models/
    subdir. Stops descending once one is found, so a models/ subfolder
    inside an experiment (there isn't one, but just in case) can't itself
    be mistaken for a separate nested experiment."""
    if (root / 'models').is_dir():
        return [root]
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if (p / 'models').is_dir():
            found.append(p)
            dirnames[:] = []
    return found


def checkpoints_to_skip(exp_dir: Path, keep_n: int) -> set:
    models_dir = exp_dir / 'models'
    if not models_dir.is_dir() or keep_n <= 0:
        return set()
    checkpoints = sorted(
        models_dir.glob('checkpoint_*.tar'),
        key=lambda p: p.stat().st_mtime)
    if len(checkpoints) <= keep_n:
        return set()
    return set(checkpoints[:-keep_n])


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
        exp_dirs = discover_experiment_dirs(root)
        if not exp_dirs:
            print(f"skipping {root}: no experiment (dir with a models/ "
                  "subdir) found under it")
            continue
        for exp_dir in exp_dirs:
            skip = checkpoints_to_skip(exp_dir, keep_n)
            n_checkpoints_skipped += len(skip)
            for p in exp_dir.rglob('*'):
                if p in skip:
                    continue
                if p.is_symlink():
                    # Not dereferenced -- matches `du`'s default (no
                    # --dereference) and avoids silently pulling a
                    # symlink's target (which may live entirely outside
                    # exp_dir, e.g. a shared init checkpoint) into the
                    # archive.
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
                         'recursively for experiment folders')
    parser.add_argument('-o', '--out', type=Path, required=True,
                         help='output .zip path')
    parser.add_argument('--keep-n', type=int, default=10,
                         help='most recent checkpoints to keep per exp '
                         '(default: 10); <= 0 keeps every checkpoint')
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
