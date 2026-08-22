#!/usr/bin/env python3
# Cross-checks every `output_path:` declared across models/10attractors/**/*.yaml
# against what's actually present locally under experiments/10attractors/, so
# you can re-run this any time to see which training runs still haven't been
# synced down from the server (rather than eyeballing configs by hand).
#
# Usage:
#   python scripts/check_experiment_sync.py [--models-root DIR] [--experiments-root DIR]
#       [--server-prefix PREFIX] [--verbose]
#
# By default, models-root/experiments-root are resolved relative to this
# script's own repo root (parent of scripts/). Override them if you're
# running from a worktree where experiments/ isn't checked out locally --
# point --experiments-root at the main checkout's experiments/10attractors
# instead (experiments/ is gitignored, so worktrees never have it).
#
# For each `output_path:` value found (server-absolute paths like
# /data/ocr/namvt17/custom-diaper/experiments/10attractors/<name>[/<subdir>],
# or relative ones like experiments/10attractors/<name>/<subdir> -- both
# forms appear across the repo's configs), this checks whether
# <output_path>/models/checkpoint_*.tar exists locally (matching how
# train.py actually writes checkpoints -- see CLAUDE.md "Outputs"). Configs
# with an unfilled `<output directory>` placeholder are silently skipped.
#
# At the end, prints a ready-to-copy `scripts/zip_experiments.py` command
# (server-side paths) covering every experiment root that isn't fully
# synced yet, so you don't have to hand-assemble it again.

import argparse
import re
from pathlib import Path

OUTPUT_PATH_RE = re.compile(r'^\s*output_path:\s*(.+?)\s*$')
EXPERIMENTS_MARKER = 'experiments/'


def extract_relative_path(raw_value: str):
    value = raw_value.strip().strip('"\'')
    # Drop a trailing inline comment, e.g. "...maximum5spks #<output directory>"
    value = value.split('#', 1)[0].strip()
    value = value.replace('\\', '/')
    idx = value.find(EXPERIMENTS_MARKER)
    if idx == -1:
        return None
    return value[idx:]


def top_level_of(relative_path: str):
    parts = relative_path.split('/')
    if len(parts) >= 3 and parts[0] == 'experiments':
        return '/'.join(parts[:3])
    return relative_path


def find_output_paths(models_root: Path, verbose: bool):
    """Yields (config_file, relative_path) for every output_path: line
    found under models_root, in declaration order."""
    for yaml_file in sorted(models_root.rglob('*.yaml')):
        for line in yaml_file.read_text(encoding='utf-8').splitlines():
            m = OUTPUT_PATH_RE.match(line)
            if not m:
                continue
            rel = extract_relative_path(m.group(1))
            if rel is None:
                if verbose:
                    print(f"  skipping unrecognized output_path in "
                          f"{yaml_file}: {m.group(1)!r}")
                continue
            yield yaml_file, rel


def checkpoint_count(repo_root: Path, relative_path: str):
    models_dir = repo_root / relative_path / 'models'
    if not models_dir.is_dir():
        return None  # directory doesn't exist at all
    return len(list(models_dir.glob('checkpoint_*.tar')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--models-root', type=Path,
                         default=default_repo_root / 'models' / '10attractors',
                         help='directory to scan for *.yaml configs '
                         '(default: <repo>/models/10attractors)')
    parser.add_argument('--experiments-root', type=Path,
                         default=default_repo_root,
                         help='repo root that experiments/10attractors/ '
                         'hangs off of -- override this to point at the '
                         'main checkout when running from a worktree '
                         '(default: this script\'s own repo root)')
    parser.add_argument('--server-prefix', type=str,
                         default='/data/ocr/namvt17/custom-diaper/',
                         help='prefix to reconstruct server-absolute paths '
                         'in the suggested zip_experiments.py command '
                         '(default: %(default)s)')
    parser.add_argument('--verbose', action='store_true',
                         help='also print output_path lines that could not '
                         'be parsed as an experiments/ path')
    args = parser.parse_args()

    entries = list(find_output_paths(args.models_root, args.verbose))
    if not entries:
        print(f"no output_path: entries found under {args.models_root}")
        return

    # top_level -> list of (config_file, relative_path, n_checkpoints|None)
    by_top_level = {}
    for config_file, rel in entries:
        n = checkpoint_count(args.experiments_root, rel)
        top = top_level_of(rel)
        by_top_level.setdefault(top, []).append((config_file, rel, n))

    incomplete_top_levels = []
    for top in sorted(by_top_level):
        stages = by_top_level[top]
        print(f"\n{top}")
        top_incomplete = False
        for config_file, rel, n in stages:
            if n is None:
                status = "MISSING (no local dir)"
                top_incomplete = True
            elif n == 0:
                status = "EMPTY (dir exists, no checkpoints)"
                top_incomplete = True
            else:
                status = f"synced ({n} checkpoint(s))"
            print(f"  [{status}] {rel}")
            print(f"      from {config_file}")
        if top_incomplete:
            incomplete_top_levels.append(top)

    print(f"\n{len(by_top_level)} experiment root(s) referenced, "
          f"{len(incomplete_top_levels)} not fully synced locally.")

    if incomplete_top_levels:
        print("\nTo collect the missing ones on the server, run there:\n")
        print("python scripts/zip_experiments.py \\")
        for top in incomplete_top_levels:
            server_path = args.server_prefix.rstrip('/') + '/' + top
            print(f"  {server_path} \\")
        print("  -o missing_experiments.zip --keep-n 10")


if __name__ == '__main__':
    main()
