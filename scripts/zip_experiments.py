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


def zip_experiment(exp_dir: Path, out_path: Path, keep_n: int) -> None:
    skip = checkpoints_to_skip(exp_dir, keep_n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in exp_dir.rglob('*'):
            if path.is_dir() or path in skip:
                continue
            zf.write(path, path.relative_to(exp_dir.parent))
            n_files += 1
    print(f"{exp_dir.name}: wrote {out_path} "
          f"({n_files} files, skipped {len(skip)} old checkpoints)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exp_dirs', nargs='+', type=Path,
                         help='experiment folders to zip (one .zip per folder)')
    parser.add_argument('-o', '--out-dir', type=Path, required=True,
                         help='directory to write <exp_name>.zip files into')
    parser.add_argument('--keep-n', type=int, default=10,
                         help='most recent checkpoints to keep per exp '
                         '(default: 10); <= 0 keeps every checkpoint')
    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        exp_dir = exp_dir.resolve()
        if not exp_dir.is_dir():
            print(f"skipping {exp_dir}: not a directory")
            continue
        zip_experiment(exp_dir, args.out_dir / f"{exp_dir.name}.zip", args.keep_n)


if __name__ == '__main__':
    main()
