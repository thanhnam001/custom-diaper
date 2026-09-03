#!/bin/bash
# Pack the checkpoints scripts/run_5day_queue.sh needs into one archive for
# upload to the server. Run from the repo root:
#
#   ./scripts/pack_weights_for_server.sh                   # everything, ~2.8 GB
#   MODE=required ./scripts/pack_weights_for_server.sh     # warm-starts only, ~570 MB
#
# The default packs everything run_5day_queue.sh reads WITH ITS OWN
# DEFAULTS, which include RUN_DIAGNOSTICS=1 -- so the three D1 checkpoints
# are in. An earlier version defaulted to the warm-starts only, which meant
# the default pack did not satisfy the default run and D1 reported missing
# weights on the server. Use MODE=required only if you also pass
# RUN_DIAGNOSTICS=0 to the queue.
#
# Paths inside the archive are relative to experiments/10attractors, so on
# the server it unpacks straight into place:
#
#   tar -xf diaper_5day_weights.tar \
#       -C /data/ocr/namvt17/custom-diaper/experiments/10attractors
#
# NO COMPRESSION by default. These files are serialized float tensors --
# gzip spends minutes to save a few percent. Set COMPRESS=1 if the upload
# link is slower than the CPU.
#
# For the RETURN leg -- bringing the queue's results back off the server --
# use scripts/pack_results_from_server.sh.
#
# Only the last $KEEP checkpoints of each directory are packed. That is all
# anything reads: the warm-starts are loaded through init_epochs: 90-100,
# and the D1 checkpoints through MAX_CHECKPOINTS_TO_AVERAGE=10. Two of the
# source directories hold 30 checkpoints, so this is a 3x saving on them.

set -eu

EXP="${EXP:-experiments/10attractors}"
OUT="${OUT:-diaper_5day_weights.tar}"
KEEP="${KEEP:-10}"
MODE="${MODE:-all}"

# Required to start training -- the two adapt warm-starts.
REQUIRED=(
    "SC_LibriSpeech_2spk_2500h_paperlr_ebf_pretrain2500h_mlp/models"
    "SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_pretrain2500h/models"
)

# Read by the D1 diagnostic, which the queue runs BY DEFAULT
# (RUN_DIAGNOSTICS=1). Excluded only under MODE=required.
DIAGNOSTIC=(
    "SC_LibriSpeech_2spk_2500h_paperlr_ebf_adapted1-10_2500h_maximum10spks_mlp/models_finetuneRAMC/models"
    "SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_adapted1-10_2500h_maximum10spks_mlp_headoff/models_finetuneRAMC/models"
    "SC_LibriSpeech_2spk_2500h_paperlr_adapted1-10_2500h_maximum10spks/models_finetuneRAMC/models"
)

DIRS=("${REQUIRED[@]}")
if [ "$MODE" = "all" ]; then
    DIRS+=("${DIAGNOSTIC[@]}")
fi

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

if [ "${COMPRESS:-0}" = "1" ]; then
    tar -czf "$OUT" -C "$EXP" -T "$LIST"
else
    tar -cf "$OUT" -C "$EXP" -T "$LIST"
fi

echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "On the server:"
echo "  tar -xf $(basename "$OUT") \\"
echo "      -C /data/ocr/namvt17/custom-diaper/experiments/10attractors"
