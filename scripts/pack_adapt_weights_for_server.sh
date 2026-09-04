#!/bin/bash
# Pack the ADAPT-stage checkpoints that scripts/run_infer_adapt_2500h.sh
# needs into one archive for upload to the server. Run from the repo root:
#
#   ./scripts/pack_adapt_weights_for_server.sh              # ~2.0 GB
#   DRY_RUN=1 ./scripts/pack_adapt_weights_for_server.sh    # size it, pack nothing
#
# This is the adapt-stage sibling of scripts/pack_weights_for_server.sh
# (which packs the 2-speaker warm-starts the five-day queue trains FROM).
# What it packs is the other end of that pipeline: the eight 2500h
# adapt-stage output directories, i.e. every 2500h lineage's model as it
# stood after multi-speaker adaptation and BEFORE any in-domain finetuning.
# Nothing has ever been scored at that point for most of these lineages --
# the finetuned checkpoints all have RTTMs, the adapt-stage ones do not --
# so this is the input to filling that column in.
#
# Paths inside the archive are relative to experiments/10attractors, so on
# the server it unpacks straight into place, next to (not over) the
# finetune outputs already there:
#
#   tar -xf diaper_adapt_weights.tar \
#       -C /data/ocr/namvt17/custom-diaper/experiments/10attractors
#
# Only the last $KEEP (default 10) checkpoints of each directory are
# packed, because that is exactly the window run_infer_adapt_2500h.sh
# averages (MAX_CHECKPOINTS_TO_AVERAGE=10 -> --epochs 90-100). One of the
# eight directories holds 30 checkpoints, so this is a 3x saving on it.
#
# NO COMPRESSION by default -- these are serialized float tensors, gzip
# spends minutes to save a few percent. Set COMPRESS=1 if the upload link
# is slower than the CPU.

set -eu

EXP="${EXP:-experiments/10attractors}"
OUT="${OUT:-diaper_adapt_weights.tar}"
KEEP="${KEEP:-10}"
COMPRESS="${COMPRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# The eight 2500h adapt-stage output directories, in the same order
# run_infer_adapt_2500h.sh runs them. Each entry is <experiment dir>/models
# -- the adapt stage's own checkpoints, NOT models_finetune*/models.
DIRS=(
    "SC_LibriSpeech_2spk_2500h_adapted1-10_2500h_maximum10spks/models"
    "SC_LibriSpeech_2spk_2500h_paperlr_adapted1-10_2500h_maximum10spks/models"
    "SC_LibriSpeech_2spk_2500h_paperlr_ebf_adapted1-10_2500h_maximum10spks_mlp/models"
    "SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31_adapted1-10_2500h_maximum10spks_mlp/models"
    "SC_LibriSpeech_2spk_2500h_fixednoam_ebf_adapted1-10_2500h_maximum10spks_mlp/models"
    "SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_adapted1-10_2500h_maximum10spks_mlp/models"
    "SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_adapted1-10_2500h_maximum10spks_mlp_headoff/models"
    "SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_adapted1-10_2500h_maximum10spks_mlp_overlaploss3/models"
)

if [ ! -d "$EXP" ]; then
    echo "ERROR: $EXP not found. Run this from the repo root, or set EXP." >&2
    exit 1
fi

LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

total=0
for d in "${DIRS[@]}"; do
    if [ ! -d "$EXP/$d" ]; then
        echo "ERROR: missing $EXP/$d -- nothing to pack for this entry." >&2
        exit 1
    fi
    # newest $KEEP checkpoints, by epoch number not filename order
    mapfile -t picked < <(
        find "$EXP/$d" -maxdepth 1 -name 'checkpoint_*.tar' -printf '%f\n' \
        | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n | tail -n "$KEEP"
    )
    if [ "${#picked[@]}" -eq 0 ]; then
        echo "ERROR: $EXP/$d holds no checkpoint_*.tar" >&2
        exit 1
    fi
    bytes=0
    for e in "${picked[@]}"; do
        echo "$d/checkpoint_${e}.tar" >> "$LIST"
        bytes=$(( bytes + $(stat -c %s "$EXP/$d/checkpoint_${e}.tar") ))
    done
    total=$(( total + bytes ))
    echo "  ${#picked[@]} ckpt(s) ep ${picked[0]}..${picked[-1]}  $(( bytes / 1000000 )) MB  $d"
done

echo
echo "packing $(wc -l < "$LIST") files, $(( total / 1000000 )) MB -> $OUT"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing written."
    exit 0
fi

if [ "$COMPRESS" = "1" ]; then
    tar -czf "$OUT" -C "$EXP" -T "$LIST"
else
    tar -cf "$OUT" -C "$EXP" -T "$LIST"
fi

echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "On the server:"
echo "  tar -xf $(basename "$OUT") \\"
echo "      -C /data/ocr/namvt17/custom-diaper/experiments/10attractors"
echo "  ./scripts/run_infer_adapt_2500h.sh"
