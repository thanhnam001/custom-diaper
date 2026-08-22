#!/usr/bin/env python3

# Repack an existing precompute_features.py --storage-format=perfile output
# directory (one {i:08d}.pkl per chunk) into --storage-format=shard (a
# handful of shard_NNNNN.bin files + a byte-offset index), WITHOUT
# recomputing any audio features. Each chunk's already-serialized pickle
# bytes are copied verbatim into a shard, byte-for-byte -- a valid pickle
# blob copied intact is still a valid, self-delimited pickle blob, so
# nothing needs re-encoding and the result is guaranteed byte-identical
# per chunk to the original.
#
# Why you'd want this: opening ~one file per chunk per epoch is fine on
# local/NVMe disk, but on a network/parallel filesystem (NFS, Lustre, ...)
# it hits the metadata server, not disk throughput -- for a large SC corpus
# (hundreds of thousands of chunks) that shows up as epoch times that swing
# wildly depending on how loaded the shared filesystem is from other
# tenants, independent of your GPU. Sharding turns "N chunks" file opens
# into "N / chunks_per_shard" file opens per run. Both formats are read
# transparently by PrecomputedKaldiDiarizationDataset (auto-detected from
# meta.pkl), so no train/infer-side flag changes -- just point
# --*-precomputed-dir at whichever cache directory you want to use.
#
# Usage:
#   python diaper/common_utils/pack_shards.py <perfile_dir> <shard_dir> \
#       --chunks-per-shard 1000
#
# The original perfile_dir is never modified or deleted -- once you've
# checked the printed --verify-sample result, remove it yourself if you
# want the disk space back.
#
# Standalone: only needs numpy/pickle, safe to run wherever the perfile
# cache lives (no project imports required).

import argparse
import logging
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    # lets numpy<2 read caches written under numpy>=2; no-op otherwise
    import numpy2_pickle_compat  # noqa: F401
except ImportError:
    try:
        import diaper.common_utils.numpy2_pickle_compat  # noqa: F401
    except ImportError:
        pass


def _read_raw(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def _load_shard_chunk(output_dir: str, chunk_locations, idx: int):
    shard_name, offset, length = chunk_locations[idx]
    with open(os.path.join(output_dir, shard_name), 'rb') as f:
        f.seek(offset)
        return pickle.loads(f.read(length))


def _verify(input_dir: str, output_dir: str, chunk_locations, total: int,
            sample_size: int, seed: int) -> bool:
    rng = random.Random(seed)
    sample = rng.sample(range(total), min(sample_size, total))
    mismatches = []
    for idx in sample:
        with open(os.path.join(input_dir, f"{idx:08d}.pkl"), 'rb') as f:
            old = pickle.load(f)
        new = _load_shard_chunk(output_dir, chunk_locations, idx)
        same = (
            old['rec'] == new['rec'] and old['st'] == new['st']
            and old['ed'] == new['ed']
            and np.array_equal(old['Y'], new['Y'])
            and np.array_equal(old['T'], new['T'])
            and np.array_equal(old['speaker_ids'], new['speaker_ids'])
        )
        if not same:
            mismatches.append(idx)

    if mismatches:
        logging.error(
            f"VERIFY FAILED: {len(mismatches)}/{len(sample)} sampled "
            f"chunks differ between old and new dir (indices: "
            f"{mismatches[:10]}{'...' if len(mismatches) > 10 else ''}). "
            f"Do NOT delete {input_dir} -- {output_dir} is not a faithful "
            "repack.")
        return False

    logging.info(
        f"VERIFY OK: {len(sample)}/{len(sample)} sampled chunks match "
        f"exactly between {input_dir} and {output_dir}. Safe to remove "
        f"{input_dir} yourself once you're satisfied -- it is not "
        "deleted automatically.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Repack a --storage-format=perfile precompute_features.py "
                     "output directory into --storage-format=shard, without "
                     "recomputing any features.")
    parser.add_argument('input_dir', help='existing perfile precomputed_dir')
    parser.add_argument('output_dir', help='new shard-format dir to create')
    parser.add_argument('--chunks-per-shard', type=int, default=1000)
    parser.add_argument(
        '--num-read-workers', type=int, default=8,
        help="threads for reading the original chunk files. I/O-bound "
             "work releases the GIL, so threads (not processes) are "
             "enough to overlap many outstanding reads/opens against a "
             "network filesystem.")
    parser.add_argument(
        '--verify-sample', type=int, default=100,
        help="after packing, spot-check this many random chunks match "
             "between the old and new dir (0 disables). This mainly "
             "catches bugs in the packing/offset bookkeeping itself -- "
             "don't skip it on data you intend to delete the original "
             "for.")
    parser.add_argument('--seed', type=int, default=0,
                         help='RNG seed for --verify-sample chunk selection')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s %(levelname)s %(message)s')

    with open(os.path.join(args.input_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    storage_format = meta.get('storage_format', 'perfile')
    if storage_format != 'perfile':
        raise ValueError(
            f"{args.input_dir} is already storage_format={storage_format!r} "
            "-- pack_shards.py only converts perfile -> shard. If you also "
            "need to merge multiple sources, do that with "
            "merge_precomputed_features.py first, on the perfile inputs, "
            "then pack the merged result once -- that script doesn't "
            "understand shard-format dirs.")

    chunk_indices = meta['chunk_indices']
    total = len(chunk_indices)
    logging.info(f"Packing {total} chunks from {args.input_dir} into "
                 f"{args.output_dir} ({args.chunks_per_shard} chunks/shard)")

    os.makedirs(args.output_dir, exist_ok=True)

    paths = [os.path.join(args.input_dir, f"{i:08d}.pkl")
             for i in range(total)]
    chunk_locations = [None] * total
    shard_idx = -1
    count_in_shard = 0
    offset = 0
    out_f = None

    def open_next_shard():
        nonlocal shard_idx, count_in_shard, offset, out_f
        if out_f is not None:
            out_f.close()
        shard_idx += 1
        count_in_shard = 0
        offset = 0
        out_f = open(os.path.join(
            args.output_dir, f"shard_{shard_idx:05d}.bin"), 'wb')

    with ThreadPoolExecutor(max_workers=args.num_read_workers) as pool:
        for idx, blob in enumerate(pool.map(_read_raw, paths)):
            if out_f is None or count_in_shard == args.chunks_per_shard:
                open_next_shard()
            out_f.write(blob)
            chunk_locations[idx] = (
                f"shard_{shard_idx:05d}.bin", offset, len(blob))
            offset += len(blob)
            count_in_shard += 1
            done = idx + 1
            if done % 5000 == 0 or done == total:
                logging.info(f"Packed {done}/{total}")
    if out_f is not None:
        out_f.close()

    new_meta = dict(meta)
    new_meta['storage_format'] = 'shard'
    new_meta['chunks_per_shard'] = args.chunks_per_shard
    new_meta['chunk_locations'] = chunk_locations
    with open(os.path.join(args.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(new_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Done. Wrote {total} chunks into {shard_idx + 1} shard "
                 f"file(s) in {args.output_dir}. Original files in "
                 f"{args.input_dir} were NOT modified or deleted.")

    if args.verify_sample > 0:
        ok = _verify(args.input_dir, args.output_dir, chunk_locations,
                     total, args.verify_sample, args.seed)
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
