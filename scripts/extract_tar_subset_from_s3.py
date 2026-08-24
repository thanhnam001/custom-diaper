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

Two phases, meant to be run separately so the (slow, one-time) index build
is cached and reused across every subsequent subset you pull from the same
tar (e.g. once for the pretrain-phase pull, reused for the adapt-phase pull):

    1. index   -- walks every header in the tar (one small Range GET per
                  member) and caches {name, offset, size} for every regular
                  file to a local JSON file. Also prints a per-top-level-
                  directory summary so you can sanity-check counts before
                  deciding what fraction to keep.
    2. extract -- loads the cached index, picks an evenly-spread subset
                  (stride sampling, not just a prefix -- see rationale in
                  `select_subset`) per top-level directory, and downloads
                  only those members' exact byte ranges to --dest-dir.

Example
-------
    # 1) One-time index build (slow: one request per member in the tar).
    python scripts/extract_tar_subset_from_s3.py index \\
        --endpoint-url https://s3-b200.internal.example \\
        --bucket ttnt-data --key ocr/namvt17/diaper_2500h_fixed_2spks.tar \\
        --index-cache /data/cache/diaper_2500h.index.json

    # 2) Pull an evenly-spread ~12% (300h out of 2500h) subset.
    python scripts/extract_tar_subset_from_s3.py extract \\
        --endpoint-url https://s3-b200.internal.example \\
        --bucket ttnt-data --key ocr/namvt17/diaper_2500h_fixed_2spks.tar \\
        --index-cache /data/cache/diaper_2500h.index.json \\
        --fraction 0.12 --dest-dir /data/subsets/diaper_300h_pretrain

Credentials/endpoint come from the standard boto3 chain (env vars
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or --profile).
If you already have this bucket configured as an rclone remote (e.g.
`s3-b200`), you can read its endpoint/keys with `rclone config show
s3-b200` and pass them via --endpoint-url / the AWS_* env vars instead of
duplicating them in a new rclone config.
"""

import argparse
import io
import json
import os
import tarfile

import boto3
from botocore.config import Config


class S3RangeFile(io.RawIOBase):
    """Seekable read-only file-like object backed by S3 Range GETs.

    Seeking is free (just moves a cursor); only .read() issues a request,
    fetching exactly the bytes asked for. Handed to `tarfile`, this makes
    its normal seek-past-data-to-next-header walk pull only 512-byte header
    reads instead of the whole object.
    """

    def __init__(self, client, bucket, key, size):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        return self._pos

    def readinto(self, b):
        if self._pos >= self._size:
            return 0
        n = len(b)
        end = min(self._pos + n, self._size) - 1
        resp = self._client.get_object(
            Bucket=self._bucket, Key=self._key,
            Range=f"bytes={self._pos}-{end}",
        )
        data = resp["Body"].read()
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


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


def build_index(args):
    client = make_client(args)
    size = client.head_object(Bucket=args.bucket, Key=args.key)["ContentLength"]
    print(f"tar object size: {size / 1e9:.2f} GB")

    s3file = S3RangeFile(client, args.bucket, args.key, size)
    entries = []
    with tarfile.open(fileobj=s3file, mode="r:") as tf:
        for i, member in enumerate(tf):
            if member.isfile():
                entries.append(
                    {"name": member.name, "offset": member.offset_data, "size": member.size}
                )
            if (i + 1) % 5000 == 0:
                print(f"  ...indexed {i + 1} members ({s3file.tell() / 1e9:.2f} GB scanned)")

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
    for i, e in enumerate(selected):
        dest_path = os.path.join(args.dest_dir, e["name"])
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == e["size"]:
            continue  # resumable: skip files already pulled with the right size
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        end = e["offset"] + e["size"] - 1
        resp = client.get_object(
            Bucket=args.bucket, Key=args.key, Range=f"bytes={e['offset']}-{end}"
        )
        with open(dest_path, "wb") as out:
            out.write(resp["Body"].read())
        if (i + 1) % 500 == 0:
            print(f"  ...downloaded {i + 1}/{len(selected)}")
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
    p_index.set_defaults(func=build_index)

    p_extract = sub.add_parser("extract", parents=[common], help="download a subset using a cached index")
    p_extract.add_argument("--dest-dir", required=True)
    p_extract.add_argument("--fraction", type=float, required=True, help="fraction of members to keep per top-level dir, e.g. 0.12 for 300h out of 2500h")
    p_extract.add_argument("--include-prefix", action="append", default=None, help="only consider these top-level dirs (repeatable); default: all")
    p_extract.add_argument("--dry-run", action="store_true", help="print what would be downloaded without downloading")
    p_extract.set_defaults(func=extract_subset)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.func(args)
