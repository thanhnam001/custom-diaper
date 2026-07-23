#!/usr/bin/env python3

# Build a Kaldi-style data dir (wav.scp, segments, utt2spk, spk2utt,
# reco2dur, rttm) for one split, out of a folder of wavs plus RTTM
# annotations. This replaces examples/prepare_data_dir.sh (which shells
# out to Kaldi utils/ scripts we don't have here) for corpora whose RTTMs
# don't come as "one file per recording named after a supplied list", but
# as one of:
#
#   --rttm-file PATH   a single RTTM containing every recording of the
#                      split (e.g. MSDWild's few.train/few.val/many.val
#                      RTTMs)
#   --rttm-dir  DIR    one *.rttm file per recording, all belonging to
#                      the split (e.g. RAMC's train/dev/test folders)
#
# Recording ids are taken from the RTTM content itself (the <uri> field),
# not from filenames, since a directory of per-recording RTTMs need not
# name files after the recording id.
#
# Output feeds directly into common_utils/precompute_features.py, which
# only reads wav.scp/segments/utt2spk/spk2utt/reco2dur (not rttm -- rttm
# is written too, for scoring/reference use with dscore etc).

import argparse
import glob
import os

import soundfile as sf


def read_rttm_lines(path: str) -> list[tuple[str, str, float, float]]:
    turns = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != 'SPEAKER':
                continue
            rec = fields[1]
            start = float(fields[3])
            dur = float(fields[4])
            spk = fields[7]
            if dur <= 0:
                continue
            turns.append((rec, spk, start, dur))
    return turns


def collect_turns(
    rttm_file: str | None, rttm_dir: str | None
) -> list[tuple[str, str, float, float]]:
    if rttm_file:
        return read_rttm_lines(rttm_file)
    rttm_paths = sorted(glob.glob(os.path.join(rttm_dir, '*.rttm')))
    if not rttm_paths:
        raise FileNotFoundError(f"No *.rttm files found under {rttm_dir}")
    turns = []
    for p in rttm_paths:
        turns.extend(read_rttm_lines(p))
    return turns


def find_wav(wav_dir: str, recid: str, exts: list[str]) -> str | None:
    for ext in exts:
        direct = os.path.join(wav_dir, recid + ext)
        if os.path.exists(direct):
            return direct
    # recordings may be nested in subfolders (e.g. MSDWild ships per-scene
    # subdirectories) -- fall back to a recursive search.
    for ext in exts:
        matches = glob.glob(
            os.path.join(wav_dir, '**', recid + ext), recursive=True)
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav-dir', required=True,
                         help='folder containing all wav files of the '
                              'corpus (recordings for other splits are '
                              'simply ignored)')
    parser.add_argument('--rttm-file', default=None,
                         help='single RTTM file covering every recording '
                              'of this split')
    parser.add_argument('--rttm-dir', default=None,
                         help='directory with one *.rttm file per '
                              'recording of this split')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--wav-ext', default='wav,flac',
                         help='comma-separated extensions to try, in '
                              'order (default: wav,flac)')
    args = parser.parse_args()

    if (args.rttm_file is None) == (args.rttm_dir is None):
        parser.error('pass exactly one of --rttm-file or --rttm-dir')

    exts = ['.' + e.strip().lstrip('.') for e in args.wav_ext.split(',')]

    turns = collect_turns(args.rttm_file, args.rttm_dir)
    if not turns:
        raise RuntimeError('No SPEAKER turns parsed from the given RTTM(s)')

    recids = sorted({rec for rec, _, _, _ in turns})

    wav_scp = {}
    missing = []
    for rec in recids:
        wav_path = find_wav(args.wav_dir, rec, exts)
        if wav_path is None:
            missing.append(rec)
            continue
        wav_scp[rec] = os.path.abspath(wav_path)
    if missing:
        print(f"WARNING: {len(missing)} recordings referenced in the RTTM "
              f"have no matching wav under {args.wav_dir} and are dropped: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    turns = [t for t in turns if t[0] in wav_scp]
    recids = sorted(wav_scp.keys())
    if not recids:
        raise RuntimeError('No recording had a matching wav file -- '
                            'check --wav-dir/--wav-ext')

    segments = []  # (uttid, rec, start, end)
    spk2utt: dict[str, list[str]] = {}
    for rec, spk, start, dur in turns:
        end = start + dur
        spkid = f"{rec}_{spk}"
        uttid = (f"{spkid}_{int(round(start * 100)):07d}_"
                 f"{int(round(end * 100)):07d}")
        segments.append((uttid, rec, start, end, spkid))
        spk2utt.setdefault(spkid, []).append(uttid)
    segments.sort(key=lambda x: x[0])

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(args.output_dir)

    with open(os.path.join(args.output_dir, 'wav.scp'), 'w') as f:
        for rec in recids:
            # relative to wav.scp's own directory: load_wav_scp() resolves
            # relative paths against the directory containing wav.scp, so
            # this stays correct if the whole kaldi dir is moved/copied
            # elsewhere, as long as it keeps the same position relative to
            # --wav-dir.
            rel_path = os.path.relpath(wav_scp[rec], output_dir_abs)
            f.write(f"{rec} {rel_path.replace(os.sep, '/')}\n")

    with open(os.path.join(args.output_dir, 'segments'), 'w') as f:
        for uttid, rec, start, end, _ in segments:
            f.write(f"{uttid} {rec} {start:.3f} {end:.3f}\n")

    with open(os.path.join(args.output_dir, 'utt2spk'), 'w') as f:
        for uttid, _, _, _, spkid in segments:
            f.write(f"{uttid} {spkid}\n")

    with open(os.path.join(args.output_dir, 'spk2utt'), 'w') as f:
        for spkid in sorted(spk2utt):
            f.write(f"{spkid} {' '.join(sorted(spk2utt[spkid]))}\n")

    with open(os.path.join(args.output_dir, 'rttm'), 'w') as f:
        for rec, spk, start, dur in sorted(turns, key=lambda x: (x[0], x[2])):
            f.write(f"SPEAKER {rec} 1 {start:.3f} {dur:.3f} <NA> <NA> "
                    f"{spk} <NA> <NA>\n")

    with open(os.path.join(args.output_dir, 'reco2dur'), 'w') as f:
        for rec in recids:
            info = sf.info(wav_scp[rec])
            f.write(f"{rec} {info.frames / info.samplerate:.3f}\n")

    print(f"Wrote kaldi-style data dir with {len(recids)} recordings, "
          f"{len(segments)} segments, {len(spk2utt)} speakers to "
          f"{args.output_dir}")


if __name__ == '__main__':
    main()
