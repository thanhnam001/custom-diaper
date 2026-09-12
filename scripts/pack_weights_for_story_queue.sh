#!/bin/bash
# Pack the ONLY weights scripts/run_4gpu_story_queue.sh needs.
#
#   ./scripts/pack_weights_for_story_queue.sh
#   # -> diaper_story_queue_weights.tar  (~167 MB)
#
# Then on the server:
#   tar -xvf diaper_story_queue_weights.tar -C /data/ocr/namvt17/custom-diaper
#
# Paths inside the tar are already prefixed experiments/10attractors/..., so
# extracting at the repo root puts every file exactly where the story-queue
# configs expect it.
#
#
# WHY THERE IS ONLY ONE ITEM
# ==========================
# Arms A1-A4 train all three stages from scratch, so they need no uploaded
# weights at all -- only the precomputed feature caches, which are already on
# the server.
#
# A0 is the exception: it IS the already-trained `paperlr` lineage
# (self-attention + weighted_average + entropy term Le, at 2500h on the
# corrected Noam schedule), so it reuses paperlr's pretrain and adapt
# checkpoints instead of spending ~50 GPU-h reproducing a result we already
# have. Only its FINETUNES are re-run, because those must sit under the same
# protocol (lr 1e-5) as the other arms -- finetuning A0 at the old 1e-6 while
# A1-A4 use 1e-5 would confound every architecture comparison with a
# -1.48 DER learning-rate effect, which is larger than the effects being
# measured.
#
# KEEP=10, NOT 1. This is a FRESH-finetune queue: train.py calls
# average_checkpoints(init_model_path, init_epochs) with init_epochs 90-100,
# which needs EVERY epoch in that range present. A single latest checkpoint
# (the convention a resume-style queue uses, because train.py's resume path
# reads only the newest file by mtime) is NOT enough here and would fail.
#
#
# IF THE SERVER ALREADY HAS THESE
# ===============================
# It may -- earlier queues uploaded to the same box. Re-uploading 167 MB is
# far cheaper than debugging an arm that silently trained from random init,
# and those earlier packers ran with KEEP=1, so what is up there may be a
# single checkpoint rather than the full 90-100 range. Set SKIP_CHECK=1 to
# pack without verifying the local range is complete.

set -u

ARCHIVE="${ARCHIVE:-diaper_story_queue_weights.tar}"
SKIP_CHECK="${SKIP_CHECK:-0}"

REL="experiments/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_adapted1-10_2500h_maximum10spks/models"

cd "$(dirname "$0")/.." || exit 1

# The worktree this script may live in does not contain experiments/ (it is
# gitignored, so it exists only in the main checkout). Resolve to the main
# working tree if needed.
if [ ! -d "$REL" ]; then
    main_root="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
    main_root="${main_root%/.git}"
    if [ -n "$main_root" ] && [ -d "$main_root/$REL" ]; then
        echo "note: experiments/ is gitignored and absent here; using the main"
        echo "      checkout at $main_root"
        cd "$main_root" || exit 1
    fi
fi

if [ ! -d "$REL" ]; then
    echo "FATAL: $REL not found."
    echo "       A0 inherits paperlr's adapt checkpoints; without them the"
    echo "       queue's preflight will refuse to start lane B/C/D's A0 stages."
    exit 1
fi

if [ "$SKIP_CHECK" != "1" ]; then
    missing=()
    for e in $(seq 91 100); do
        [ -f "$REL/checkpoint_${e}.tar" ] || missing+=("$e")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "FATAL: paperlr adapt checkpoints missing for epoch(s): ${missing[*]}"
        echo "       train.py's average_checkpoints(90-100) needs the whole"
        echo "       range, not just the newest file. Pull them from the server"
        echo "       or an archive tar before packing."
        exit 1
    fi
    echo "ok: all 10 checkpoints (91-100) present"
fi

echo "packing $REL -> $ARCHIVE"
tar -cvf "$ARCHIVE" \
    "$REL"/checkpoint_9[1-9].tar \
    "$REL"/checkpoint_100.tar

echo
ls -lh "$ARCHIVE"
echo
echo "Upload, then on the server:"
echo "  tar -xvf $(basename "$ARCHIVE") -C /data/ocr/namvt17/custom-diaper"
echo
echo "Then verify with:"
echo "  DRY_RUN=1 ./scripts/run_4gpu_story_queue.sh"
