#!/usr/bin/env python3
"""Buckets `infer.py --compute-metrics` output (metrics_per_file.csv) by the
*distinct* reference speaker count of each recording (from its RTTM), and
reports mean DER/miss/FA/confusion per bucket plus the Pearson correlation
between speaker count and each metric.

This reproduces the core analysis behind the "DER scales sharply with the
number of distinct speakers" finding in msdwild_der_gap_analysis.md (root
cause of the gap between our MSDWild reproductions and the paper's published
numbers) -- written up as a script here so it doesn't have to be
reconstructed ad hoc next time a new checkpoint needs the same breakdown.

"Distinct speakers" is counted from the RTTM directly (unique speaker names
per recording), NOT `avg_ref_spk_qty` in metrics_per_file.csv (that column
measures concurrent-overlap density -- average number of simultaneously
active speakers per frame -- not speaker identity count).

Usage (single checkpoint):
    python diaper/analyze_speaker_count_gap.py \\
        --rttm database/msdwild/kaldi/test/rttm \\
        --metrics results/full_msdwild_test/mlp_unmaskeddiv/.../metrics_per_file.csv

Usage (compare multiple checkpoints side by side, e.g. ours vs the paper's):
    python diaper/analyze_speaker_count_gap.py \\
        --rttm database/msdwild/kaldi/test/rttm \\
        --metrics paper=results/paper_original_checkpoints/.../metrics_per_file.csv \\
        --metrics ours=results/full_msdwild_test/mlp_unmaskeddiv/.../metrics_per_file.csv

Bucket edges default to {2, 3, 4+} (MSDWild's speaker-count range) --
override with --bucket-edges for other datasets (e.g. RAMC has more
speakers per conversation).
"""
import argparse
import csv
from collections import defaultdict


def load_speaker_counts(rttm_path: str) -> dict:
    """recording_id -> number of distinct speaker names in its RTTM lines."""
    speakers_per_reco = defaultdict(set)
    with open(rttm_path) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            reco, spk_name = fields[1], fields[7]
            speakers_per_reco[reco].add(spk_name)
    return {reco: len(spks) for reco, spks in speakers_per_reco.items()}


def load_metrics(metrics_csv_path: str) -> dict:
    """recording_id -> dict of float metric columns."""
    rows = {}
    with open(metrics_csv_path, newline='') as f:
        for row in csv.DictReader(f):
            name = row.pop('name')
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = float('nan')
            rows[name] = parsed
    return rows


def bucket_label(n_speakers: int, edges: list) -> str:
    for edge in edges[:-1]:
        if n_speakers == edge:
            return str(edge)
    return f"{edges[-1]}+"


def pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return float('nan')
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float('nan')
    return cov / (var_x * var_y) ** 0.5


def analyze(label: str, speaker_counts: dict, metrics: dict, edges: list,
            metric_keys: list) -> None:
    joined = []
    missing = 0
    for reco, n_spk in speaker_counts.items():
        if reco not in metrics:
            missing += 1
            continue
        joined.append((n_spk, metrics[reco]))
    if missing:
        print(f"  [{label}] {missing} recordings in RTTM had no matching "
              f"row in metrics_per_file.csv (skipped)")

    buckets = defaultdict(list)
    for n_spk, m in joined:
        buckets[bucket_label(n_spk, edges)].append(m)

    print(f"\n=== {label} ({len(joined)} files) ===")
    header = "n_spk".ljust(8) + "count".rjust(7) + "".join(
        k.rjust(12) for k in metric_keys)
    print(header)
    for key in sorted(buckets, key=lambda k: (len(k), k)):
        rows = buckets[key]
        line = key.ljust(8) + str(len(rows)).rjust(7)
        for mk in metric_keys:
            vals = [r[mk] for r in rows if r[mk] == r[mk]]  # drop NaN
            mean = sum(vals) / len(vals) if vals else float('nan')
            line += f"{mean:.2f}".rjust(12)
        print(line)

    all_n = [n for n, _ in joined]
    print("correlation(n_speakers, metric):")
    for mk in metric_keys:
        vals = [(n, m[mk]) for n, m in joined if m[mk] == m[mk]]
        if len(vals) < 2:
            continue
        xs, ys = zip(*vals)
        print(f"  {mk}: r={pearson(list(xs), list(ys)):+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rttm', required=True,
                         help="Reference RTTM (e.g. database/msdwild/kaldi/test/rttm)")
    parser.add_argument('--metrics', required=True, action='append',
                         help="Path to metrics_per_file.csv, optionally "
                              "prefixed 'label=' (repeatable to compare "
                              "checkpoints side by side)")
    parser.add_argument('--bucket-edges', type=int, nargs='+', default=[2, 3, 4],
                         help="Speaker-count bucket edges; last one becomes "
                              "an open-ended 'N+' bucket (default: 2 3 4, "
                              "matching MSDWild's range)")
    parser.add_argument('--metric-keys', nargs='+',
                         default=['DER', 'DER_miss', 'DER_FA', 'DER_conf'],
                         help="Columns from metrics_per_file.csv to report "
                              "(default: DER DER_miss DER_FA DER_conf)")
    args = parser.parse_args()

    speaker_counts = load_speaker_counts(args.rttm)
    print(f"Loaded {len(speaker_counts)} recordings from {args.rttm}")

    for spec in args.metrics:
        if '=' in spec:
            label, path = spec.split('=', 1)
        else:
            label, path = spec, spec
        metrics = load_metrics(path)
        analyze(label, speaker_counts, metrics, args.bucket_edges,
                args.metric_keys)


if __name__ == '__main__':
    main()
