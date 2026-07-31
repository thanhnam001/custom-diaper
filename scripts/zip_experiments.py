#!/usr/bin/env python3
# Zip experiment folders for transfer, keeping only the N most recent
# checkpoints per experiment (tensorboard logs / configs / everything else
# under the exp folder is zipped unchanged). Originals are never modified.
#
# Usage:
#   python scripts/zip_experiments.py EXP_DIR [EXP_DIR ...] -o OUT_DIR [--keep-n 10]
#
# Each EXP_DIR becomes OUT_DIR/<exp_name>.zip. Checkpoints under
# <EXP_DIR>/models/checkpoint_*.tar are ranked by mtime (mirrors the sort
# train.py itself uses to find the latest checkpoint, see train.py's
# `paths.sort(key=lambda x: os.path.getmtime(x))`); only the --keep-n
# newest are included in the archive.

import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm


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


def zip_experiment(
    exp_dir: Path, out_path: Path, keep_n: int, compress_level
) -> None:
    skip = checkpoints_to_skip(exp_dir, keep_n)
    files = [p for p in exp_dir.rglob('*') if p.is_file() and p not in skip]
    total_bytes = sum(p.stat().st_size for p in files)

    if compress_level is None:
        zip_kwargs = dict(compression=zipfile.ZIP_STORED)
    else:
        zip_kwargs = dict(
            compression=zipfile.ZIP_DEFLATED, compresslevel=compress_level)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', **zip_kwargs) as zf, \
            tqdm(total=total_bytes, unit='B', unit_scale=True,
                 desc=exp_dir.name) as pbar:
        for path in files:
            zf.write(path, path.relative_to(exp_dir.parent))
            pbar.update(path.stat().st_size)
    print(f"{exp_dir.name}: wrote {out_path} "
          f"({len(files)} files, skipped {len(skip)} old checkpoints)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exp_dirs', nargs='+', type=Path,
                         help='experiment folders to zip (one .zip per folder)')
    parser.add_argument('-o', '--out-dir', type=Path, required=True,
                         help='directory to write <exp_name>.zip files into')
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

    for exp_dir in args.exp_dirs:
        exp_dir = exp_dir.resolve()
        if not exp_dir.is_dir():
            print(f"skipping {exp_dir}: not a directory")
            continue
        zip_experiment(
            exp_dir, args.out_dir / f"{exp_dir.name}.zip",
            args.keep_n, args.compress_level)


if __name__ == '__main__':
    main()
