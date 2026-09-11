#!/bin/bash
# Pack exactly the weights scripts/run_4gpu_msdwild_queue.sh needs, and
# nothing else. Run from the repo root:
#
#   ./scripts/pack_weights_for_4gpu_msdwild_queue.sh
#   # -> diaper_4gpu_msdwild_weights.tar  (~594 MB)
#
# Then on the server:
#   tar -xvf diaper_4gpu_msdwild_weights.tar -C /data/ocr/namvt17/custom-diaper
#
# The tar stores paths relative to the repo root already prefixed with
# experiments/10attractors/..., so extracting at the repo root puts every
# file exactly where the queue's configs expect it.
#
#
# WHAT IS ALREADY ON THE SERVER, AND WHY THIS STILL SHIPS SOME OF IT
# ===================================================================
# The 36h resume queue and the pos_weight queue both uploaded weights to this
# same box, so several of these directories may already exist there. This
# packer still includes them by default, because:
#   - the 36h queue's packer ran with KEEP=1 (latest checkpoint only), so
#     what is up there is a MOVING target that later rounds overwrote;
#   - lanes B and C need the checkpoint at a SPECIFIC epoch (750 / 656) that
#     the newest local pull confirms, not "whatever is up there";
#   - re-uploading ~170 MB is far cheaper than debugging a lane that silently
#     resumed from the wrong epoch or trained from scratch.
# Use the SKIP_* flags below if you have verified what is on the server.
#
# Per-item status (checked 2026-09-11):
#
#   1. PAPER adapt 91-100        DEFINITELY NOT on the server. It comes from
#      (167 MB, 10 ckpts)        the upstream read-only checkout at
#                                ../Master/repos/DiaPer, has never been part
#                                of any previous packer, and lane A cannot
#                                run without it. This is the one item you
#                                must not skip.
#
#   2. conformer_k31 ADAPT       MAY already be there (the pos_weight queue's
#      91-100 (257 MB)           preflight needed an adapt dir, and the 36h
#                                queue's resume lanes referenced it). Lane D
#                                is a FRESH finetune, so it needs all 10
#                                epochs for average_checkpoints(90-100) --
#                                a single latest checkpoint is NOT enough.
#                                SKIP_CNF_ADAPT=1 to omit.
#
#   3. ebf paperlr MSDWild       Lane B resume point. The 36h queue did NOT
#      ckpt 750 (90 MB)          run this stage (B3/B4 were dropped when it
#                                was narrowed to A5/A1/B2), so the server's
#                                copy may be stale or absent.
#                                SKIP_EBF_RESUME=1 to omit.
#
#   4. conformer_k31 MSDWild     Lane C resume point. This IS one of the 36h
#      ckpt 656 (80 MB)          queue's three stages (A1), so the server
#                                probably has epoch 656 already -- it is what
#                                produced the 17.06 we are extending from.
#                                Most likely of the four to be redundant.
#                                SKIP_CNF_RESUME=1 to omit.
#
# KEEP=1 vs KEEP=10 -- do not "tidy" this:
#   Resume points (3, 4) need ONE checkpoint: train.py's resume path reads
#   only the single latest checkpoint by mtime.
#   Init points (1, 2) need TEN: average_checkpoints(init_model_path,
#   "90-100") requires every epoch in that exact range and fails on a gap.
#   Shipping 1 checkpoint for an init directory is a silent
#   train-from-scratch; this is the mistake the 36h queue documented.
#
#
# ENV KNOBS
#   OUT                 output tar         (default diaper_4gpu_msdwild_weights.tar)
#   UPSTREAM            upstream DiaPer checkout (default ../Master/repos/DiaPer)
#   SKIP_CNF_ADAPT=1    omit item 2
#   SKIP_EBF_RESUME=1   omit item 3
#   SKIP_CNF_RESUME=1   omit item 4
#   DRY_RUN=1           list what would be packed, write nothing

set -u

OUT="${OUT:-diaper_4gpu_msdwild_weights.tar}"
UPSTREAM="${UPSTREAM:-../Master/repos/DiaPer}"
DRY_RUN="${DRY_RUN:-0}"

EXP=experiments/10attractors
CNF=$EXP/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31_adapted1-10_2500h_maximum10spks_mlp
EBF=$EXP/SC_LibriSpeech_2spk_2500h_paperlr_ebf_adapted1-10_2500h_maximum10spks_mlp
PAPER_SRC="$UPSTREAM/models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models"
PAPER_DST=$EXP/PAPER_SC_LibriSpeech_2spk_adapted1-10/models

[ -f diaper/train.py ] || { echo "ERROR: run from the repo root." >&2; exit 1; }

STAGE=".pack_4gpu_stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
MANIFEST="$STAGE/MANIFEST.txt"
: > "$MANIFEST"

die () { echo "ERROR: $*" >&2; rm -rf "$STAGE"; exit 1; }
note () { echo "$*"; echo "$*" >> "$MANIFEST"; }

# copy_range <srcdir> <dstdir> <label> <ep_lo> <ep_hi>   -- all epochs inclusive
copy_range () {
    local src="$1" dst="$2" label="$3" lo="$4" hi="$5" e missing=""
    [ -d "$src" ] || die "$label: source dir not found: $src"
    mkdir -p "$STAGE/$dst"
    for e in $(seq "$lo" "$hi"); do
        if [ -f "$src/checkpoint_${e}.tar" ]; then
            cp "$src/checkpoint_${e}.tar" "$STAGE/$dst/"
        else
            missing="$missing $e"
        fi
    done
    [ -n "$missing" ] && die "$label: missing checkpoints:$missing (average_checkpoints needs the full range)"
    note "  $label -> $dst  (epochs ${lo}-${hi}, $(du -sh "$STAGE/$dst" | cut -f1))"
}

# copy_latest <srcdir> <dstdir> <label>
copy_latest () {
    local src="$1" dst="$2" label="$3" last
    [ -d "$src" ] || die "$label: source dir not found: $src"
    last=$(find "$src" -maxdepth 1 -name 'checkpoint_*.tar' -exec basename {} \; \
           | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n | tail -1)
    [ -n "$last" ] || die "$label: no checkpoint_*.tar in $src"
    mkdir -p "$STAGE/$dst"
    cp "$src/checkpoint_${last}.tar" "$STAGE/$dst/"
    note "  $label -> $dst  (epoch $last only, $(du -sh "$STAGE/$dst" | cut -f1))"
}

echo "Packing weights for run_4gpu_msdwild_queue.sh"
note "diaper_4gpu_msdwild_weights -- built $(date '+%Y-%m-%d %H:%M:%S')"
note ""

# ---- 1. PAPER adapt, epochs 91-100 -- lane A, mandatory --------------------
note "[1] PAPER adapt (lane A init -- REQUIRED, never uploaded before)"
copy_range "$PAPER_SRC" "$PAPER_DST" "paper adapt" 91 100

# ---- 2. conformer_k31 adapt, epochs 91-100 -- lane D fresh finetune --------
if [ "${SKIP_CNF_ADAPT:-0}" = "1" ]; then
    note "[2] conformer_k31 adapt -- SKIPPED (SKIP_CNF_ADAPT=1)"
else
    note "[2] conformer_k31 adapt (lane D init -- needs all 10 for averaging)"
    copy_range "$CNF/models" "$CNF/models" "conformer_k31 adapt" 91 100
fi

# ---- 3. ebf paperlr MSDWild latest -- lane B resume ------------------------
if [ "${SKIP_EBF_RESUME:-0}" = "1" ]; then
    note "[3] ebf paperlr MSDWild resume -- SKIPPED (SKIP_EBF_RESUME=1)"
else
    note "[3] ebf paperlr MSDWild (lane B resume point)"
    copy_latest "$EBF/models_finetuneMSDWILD/models" "$EBF/models_finetuneMSDWILD/models" "ebf MSDWild"
fi

# ---- 4. conformer_k31 MSDWild latest -- lane C resume ----------------------
if [ "${SKIP_CNF_RESUME:-0}" = "1" ]; then
    note "[4] conformer_k31 MSDWild resume -- SKIPPED (SKIP_CNF_RESUME=1)"
else
    note "[4] conformer_k31 MSDWild (lane C resume point)"
    copy_latest "$CNF/models_finetuneMSDWILD/models" "$CNF/models_finetuneMSDWILD/models" "conformer MSDWild"
fi

note ""
note "NOT INCLUDED (deliberately):"
note "  - pos_weight arms (models_finetuneMSDWILD_posw3.0 / _posw5.0). The"
note "    lever was confirmed dead 2026-09-11 (flat dose-response between 3.0"
note "    and 5.0, and the control matches them by epoch 750 with no"
note "    reweighting), so no lane in this queue resumes them. Their weights"
note "    are already pulled locally if you ever want to revisit."
note "  - every RAMC run (A5/B2/A3/...). This queue is MSDWild-only."
note "  - A5/B2 RAMC sub5 record checkpoints -- still the most valuable"
note "    weights in the project, but not needed here."

echo
echo "----- manifest -----"
cat "$MANIFEST"
echo "--------------------"
echo "staged total: $(du -sh "$STAGE" | cut -f1)"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing written. Removing stage."
    rm -rf "$STAGE"
    exit 0
fi

cp "$MANIFEST" "$STAGE/MANIFEST.txt" 2>/dev/null || true
tar -cf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"

echo
echo "wrote $OUT ($(du -sh "$OUT" | cut -f1))"
echo
echo "On the server:"
echo "  tar -xvf $(basename "$OUT") -C /data/ocr/namvt17/custom-diaper"
echo "  cd /data/ocr/namvt17/custom-diaper"
echo "  mkdir -p logs/4gpu_msdwild_queue"
echo "  nohup ./scripts/run_4gpu_msdwild_queue.sh > logs/4gpu_msdwild_queue/driver.log 2>&1 &"
