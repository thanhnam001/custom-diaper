#!/usr/bin/env python3
"""Pull a subset of members out of a single uncompressed .tar object stored
on S3-compatible object storage (e.g. a self-hosted MinIO/Ceph bucket),
without downloading the whole archive.

Why this works
--------------
An *uncompressed* tar is just a flat sequence of 512-byte headers, each
followed by that member's data padded to a 512-byte boundary. Once a header
tells you a member's size, you know the exact byte offset the next header
starts at -- so you can walk the whole archive using only small header reads
plus pure offset arithmetic, and only pay for the actual data of the members
you choose to keep. S3 (and self-hosted S3-compatible stores) serve this via
plain HTTP Range GETs on the object, no different from reading a local file.

This does NOT work for .tar.gz/.tar.bz2/.tar.xz -- compressed streams can't
be jumped around in without decompressing everything up to that point.

At corpus sizes like 2500h (tens to hundreds of thousands of chunk files),
walking every header one Range GET at a time is latency-bound, not
bandwidth-bound: each header is only 512 bytes, but a plain tar has no
table-of-contents, so headers can only be found by walking the chain
sequentially -- 150k members means 150k round trips if done naively. The
`index` command below parallelizes this: it first finds --index-workers
confirmed header boundaries spread across the archive (each found from a
single small window read, verified via tarfile's own header checksum
validation so a false sync can't silently corrupt the index), then walks
each worker's span concurrently. This does not reduce the number of header
reads (every member is still visited once), it just spreads them across N
connections so wall-clock time drops roughly N-fold.

Two phases, meant to be run separately so the (slow-ish, one-time) index
build is cached and reused across every subsequent subset you pull from the
same tar (e.g. once for the pretrain-phase pull, reused for the adapt-phase
pull):

    1. index   -- builds the {name, offset, size} index for every regular
                  file member and caches it to a local JSON file. Also
                  prints a per-top-level-directory summary so you can
                  sanity-check counts before deciding what fraction to keep.
    2. extract -- loads the cached index, picks an evenly-spread subset
                  (stride sampling, not just a prefix -- see rationale in
                  `select_subset`) per top-level directory, and downloads
                  only those members' exact byte ranges to --dest-dir,
                  concurrently (these downloads have no ordering
                  dependency on each other, unlike the header walk).

Example
-------
    # 1) One-time index build.
    python scripts/extract_tar_subset_from_s3.py index \\
        --endpoint-url https://s3-b200.internal.example \\
        --bucket ttnt-data --key ocr/namvt17/diaper_2500h_fixed_2spks.tar \\
        --index-cache /data/cache/diaper_2500h.index.json --index-workers 16

    # 2) Pull an evenly-spread ~12% (300h out of 2500h) subset.
    python scripts/extract_tar_subset_from_s3.py extract \\
        --endpoint-url https://s3-b200.internal.example \\
        --bucket ttnt-data --key ocr/namvt17/diaper_2500h_fixed_2spks.tar \\
        --index-cache /data/cache/diaper_2500h.index.json \\
        --fraction 0.12 --dest-dir /data/subsets/diaper_300h_pretrain --concurrency 16

Credentials/endpoint come from the standard boto3 chain (env vars
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or --profile).
If you already have this bucket configured as an rclone remote (e.g.
`s3-b200`), you can read its endpoint/keys with `rclone config show
s3-b200` and pass them via --endpoint-url / the AWS_* env vars instead of
duplicating them in a new rclone config.

Assumes plain (POSIX/GNU short-name) tar entries, i.e. no GNU long-name
(">100 char path) extension headers -- true for the fixed-width, short,
zero-padded chunk filenames this precompute cache format uses
(train/00000000.pkl, ...). If you point this at a tar with long paths, the
manual per-worker header walk below (`walk_region`) would misparse them;
the `index` command's file-count summary vs. what you expect is a quick way
to notice if that happened.
"""

import argparse
import concurrent.futures
import json
import os
import tarfile

import boto3
from botocore.config import Config

HEADER_SIZE = 512


def make_client(args):
    return boto3.client(
        "s3",
        endpoint_url=args.endpoint_url,
        aws_access_key_id=args.access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=args.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=args.region,
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10}),
        verify=not args.no_verify_ssl,
    )


def fetch_range(client, bucket, key, start, end):
    """Inclusive byte range [start, end]."""
    resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def parse_header(buf):
    """Returns a tarfile.TarInfo for a candidate 512-byte header, or None if
    it isn't a valid header (wrong checksum, wrong magic, etc.) -- reuses
    tarfile's own validation rather than reimplementing the checksum
    algorithm (which has GNU/POSIX signed-vs-unsigned edge cases)."""
    if len(buf) < HEADER_SIZE or buf == b"\0" * HEADER_SIZE:
        return None
    try:
        return tarfile.TarInfo.frombuf(buf, tarfile.ENCODING, "surrogateescape")
    except tarfile.TarError:
        return None


def resync_to_header(client, bucket, key, approx_pos, total_size, window=4 * 1024 * 1024):
    """Find the offset of a real tar header at or after approx_pos.

    Fetches one local window and scans 512-byte-aligned candidates within
    it for a header whose checksum validates *and* whose implied next
    header also validates -- this two-header chain check makes a
    coincidental false-positive (garbage pickle bytes that happen to
    checksum-match) astronomically unlikely, since real corruption would
    have to fool the check twice in a row.
    """
    start = (approx_pos // HEADER_SIZE) * HEADER_SIZE
    end = min(start + window, total_size)
    buf = fetch_range(client, bucket, key, start, end - 1)

    for off in range(0, len(buf) - HEADER_SIZE, HEADER_SIZE):
        candidate = buf[off : off + HEADER_SIZE]
        info = parse_header(candidate)
        if info is None or not info.name:
            continue
        abs_offset = start + off
        next_offset = abs_offset + HEADER_SIZE + ((info.size + HEADER_SIZE - 1) // HEADER_SIZE) * HEADER_SIZE
        if next_offset >= total_size:
            return abs_offset  # near EOF, trust single validation
        if next_offset < end:
            next_buf = buf[next_offset - start : next_offset - start + HEADER_SIZE]
        else:
            next_buf = fetch_range(client, bucket, key, next_offset, next_offset + HEADER_SIZE - 1)
        if parse_header(next_buf) is not None:
            return abs_offset
    raise RuntimeError(
        f"could not resync to a tar header within {window} bytes of offset {approx_pos}; "
        "try a larger --resync-window"
    )


def walk_region(client, bucket, key, start, end, total_size):
    """Sequentially walk headers in [start, end), collecting regular files.
    May run slightly past `end` to finish the member straddling the
    boundary; the next worker's confirmed start is always a real header
    start so there is no double-counting."""
    entries = []
    pos = start
    while pos < end and pos < total_size:
        buf = fetch_range(client, bucket, key, pos, pos + HEADER_SIZE - 1)
        info = parse_header(buf)
        if info is None:
            break  # end-of-archive marker (or truncated tail)
        data_offset = pos + HEADER_SIZE
        if info.isfile():
            entries.append({"name": info.name, "offset": data_offset, "size": info.size})
        pos = data_offset + ((info.size + HEADER_SIZE - 1) // HEADER_SIZE) * HEADER_SIZE
    return entries


def build_index(args):
    client = make_client(args)
    total_size = client.head_object(Bucket=args.bucket, Key=args.key)["ContentLength"]
    print(f"tar object size: {total_size / 1e9:.2f} GB")

    k = max(1, args.index_workers)
    boundaries = [0]
    for i in range(1, k):
        approx = total_size * i // k
        confirmed = resync_to_header(client, args.bucket, args.key, approx, total_size, args.resync_window)
        boundaries.append(confirmed)
    boundaries.append(total_size)
    # dedupe in case two approx points resynced to the same header on a small archive
    boundaries = sorted(set(boundaries))

    print(f"walking {len(boundaries) - 1} regions concurrently...")
    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=k) as pool:
        futures = [
            pool.submit(walk_region, client, args.bucket, args.key, boundaries[i], boundaries[i + 1], total_size)
            for i in range(len(boundaries) - 1)
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            region_entries = fut.result()
            entries.extend(region_entries)
            print(f"  ...region {i + 1}/{len(futures)} done ({len(region_entries)} files)")

    entries.sort(key=lambda e: e["offset"])
    with open(args.index_cache, "w") as f:
        json.dump(entries, f)
    print(f"wrote index for {len(entries)} files to {args.index_cache}")

    summary_by_prefix(entries)


def summary_by_prefix(entries):
    counts, sizes = {}, {}
    for e in entries:
        top = e["name"].split("/", 1)[0]
        counts[top] = counts.get(top, 0) + 1
        sizes[top] = sizes.get(top, 0) + e["size"]
    print("summary by top-level directory:")
    for top in sorted(counts):
        print(f"  {top}: {counts[top]} files, {sizes[top] / 1e9:.2f} GB")


def select_subset(entries, fraction, prefixes):
    """Stride-sample `fraction` of entries under each prefix, preserving
    the original ordering position (not just the first `fraction`).

    A contiguous prefix of the archive (e.g. its first 12%) risks being
    drawn from a narrow slice of whatever source pool the simulated-
    conversation generator iterated through first, which can under-
    represent the speaker/utterance diversity the full corpus has. Evenly
    striding across the whole ordered list keeps the same spread the full
    dataset has, just thinned out.
    """
    by_prefix = {}
    for e in entries:
        top = e["name"].split("/", 1)[0]
        if prefixes and top not in prefixes:
            continue
        by_prefix.setdefault(top, []).append(e)

    selected = []
    for top, items in by_prefix.items():
        items.sort(key=lambda e: e["name"])
        n_keep = max(1, round(len(items) * fraction))
        stride = len(items) / n_keep
        picked = [items[int(i * stride)] for i in range(n_keep)]
        print(f"  {top}: keeping {len(picked)}/{len(items)} (stride {stride:.2f})")
        selected.extend(picked)
    return selected


def download_one(client, bucket, key, dest_dir, entry):
    dest_path = os.path.join(dest_dir, entry["name"])
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == entry["size"]:
        return  # resumable: skip files already pulled with the right size
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    end = entry["offset"] + entry["size"] - 1
    data = fetch_range(client, bucket, key, entry["offset"], end)
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as out:
        out.write(data)
    os.replace(tmp_path, dest_path)


def extract_subset(args):
    with open(args.index_cache) as f:
        entries = json.load(f)

    prefixes = set(args.include_prefix) if args.include_prefix else None
    selected = select_subset(entries, args.fraction, prefixes)
    total_bytes = sum(e["size"] for e in selected)
    print(f"selected {len(selected)} files, {total_bytes / 1e9:.2f} GB to download")

    if args.dry_run:
        return

    client = make_client(args)
    os.makedirs(args.dest_dir, exist_ok=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(download_one, client, args.bucket, args.key, args.dest_dir, e) for e in selected
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # re-raise any download error instead of silently dropping it
            done += 1
            if done % 500 == 0:
                print(f"  ...processed {done}/{len(selected)} (existing files with a matching size are skipped)")
    print(f"done: {len(selected)} files written under {args.dest_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint-url", required=True, help="S3-compatible endpoint URL")
    common.add_argument("--bucket", required=True)
    common.add_argument("--key", required=True, help="path to the .tar object within the bucket")
    common.add_argument("--access-key", default=None)
    common.add_argument("--secret-key", default=None)
    common.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"), help="required for SigV4 signing even against self-hosted S3 that ignores its value")
    common.add_argument("--no-verify-ssl", action="store_true")
    common.add_argument("--index-cache", required=True, help="local JSON path to write/read the member index")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", parents=[common], help="scan the tar's headers and cache the member index")
    p_index.add_argument("--index-workers", type=int, default=8, help="concurrent header-walk workers; keep modest if the self-hosted store rate-limits (default: 8)")
    p_index.add_argument("--resync-window", type=int, default=4 * 1024 * 1024, help="bytes fetched to locate a real header near each worker boundary (default: 4MB)")
    p_index.set_defaults(func=build_index)

    p_extract = sub.add_parser("extract", parents=[common], help="download a subset using a cached index")
    p_extract.add_argument("--dest-dir", required=True)
    p_extract.add_argument("--fraction", type=float, required=True, help="fraction of members to keep per top-level dir, e.g. 0.12 for 300h out of 2500h")
    p_extract.add_argument("--include-prefix", action="append", default=None, help="only consider these top-level dirs (repeatable); default: all")
    p_extract.add_argument("--concurrency", type=int, default=16, help="concurrent download workers (default: 16)")
    p_extract.add_argument("--dry-run", action="store_true", help="print what would be downloaded without downloading")
    p_extract.set_defaults(func=extract_subset)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.func(args)
