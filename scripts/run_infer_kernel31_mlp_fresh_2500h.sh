#!/bin/bash
set -e

# Runs inference (+ dscore scoring) for the 2 finetuned checkpoints of the
# SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h lineage
# (models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h/):
#   - MSDWild finetune: infer_msdwild_mlp.yaml
#   - RAMC finetune:    infer_ramc_mlp.yaml
#
# For SERVER use: those 2 configs' infer_data_dir/models_path/rttms_dir are
# already correct absolute paths on the training server's filesystem, so
# this script reads and uses them as-is -- no path rewriting, unlike
# scripts/rerun_infer_msdwild.sh which translates server paths to a local
# mirror for running off-server. Only `epochs` is filled in here (omitted
# from the configs since no checkpoints existed anywhere when they were
# written): this script auto-detects it from whatever checkpoint_<N>.tar
# files exist under each config's models_path, averaging the last
# $MAX_CHECKPOINTS_TO_AVERAGE of them ("N-1 to M" parse_epochs syntax, see
# diaper/backend/models.py::parse_epochs). There's no dev-DER curve to
# hand-pick a better range from yet -- redo the run with an explicit
# --epochs once TensorBoard shows one.
#
# A config is skipped if its models_path has no checkpoints yet.
#
# Reference RTTM for scoring is derived from each config's own
# infer_data_dir (Kaldi data dirs bundle their own `rttm` file).
#
# Uses the SERVER's diaper/dscore envs and checkouts (same values as
# scripts/run_infer_kernel31_mlp_fresh.sh) -- NOT this repo's local
# ../Master/repos/DiaPer/.diaper_env / ../Master/repos/dscore paths, which
# only exist on the Windows dev machine.
#
# Run from the repo root, on the server:
#   ./scripts/run_infer_kernel31_mlp_fresh_2500h.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAPER_ENV="/data/ocr/namvt17/custom-diaper/.venv"
DSCORE_SRC="/data/ocr/namvt17/custom-diaper/dscore"
DSCORE_ENV="/data/ocr/namvt17/custom-diaper/dscore/.dscore"

MAX_CHECKPOINTS_TO_AVERAGE=10

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h"

# name|config|dscore collar (empty = no --collar flag, i.e. 0s)
RUNS=(
    "msdwild_mlp|${CONFIG_DIR}/infer_msdwild_mlp.yaml|0.25"
    "ramc_mlp|${CONFIG_DIR}/infer_ramc_mlp.yaml|"
)

for run in "${RUNS[@]}"; do
    IFS='|' read -r name cfg collar <<< "$run"

    models_path=$(grep "^models_path:" "$cfg" | sed 's/^models_path: *//')
    rttms_dir=$(grep "^rttms_dir:" "$cfg" | sed 's/^rttms_dir: *//')
    infer_data_dir=$(grep "^infer_data_dir:" "$cfg" | sed 's/^infer_data_dir: *//')
    ref_rttm="${infer_data_dir}/rttm"

    if [ ! -d "$models_path" ]; then
        echo "SKIPPING $name ($cfg) -- no checkpoints found at $models_path"
        continue
    fi

    mapfile -t ckpt_epochs < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
        -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
        echo "SKIPPING $name ($cfg) -- $models_path exists but has no checkpoint_*.tar files"
        continue
    fi

    last_idx=$(( ${#ckpt_epochs[@]} - 1 ))
    last_epoch="${ckpt_epochs[$last_idx]}"
    start_idx=$(( ${#ckpt_epochs[@]} > MAX_CHECKPOINTS_TO_AVERAGE ? ${#ckpt_epochs[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
    first_epoch_averaged="${ckpt_epochs[$start_idx]}"
    # parse_epochs' "-" syntax averages (start, end] -- subtract 1 so the
    # first epoch in ckpt_epochs is actually included.
    epochs_range="$(( first_epoch_averaged - 1 ))-${last_epoch}"

    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] running: $name ($cfg)"
    echo "  infer_data_dir: $infer_data_dir"
    echo "  models_path:    $models_path"
    echo "  rttms_dir:      $rttms_dir"
    echo "  epochs:         $epochs_range (averaging ${#ckpt_epochs[@]} available checkpoint(s), capped at $MAX_CHECKPOINTS_TO_AVERAGE)"
    echo "=================================================================="

    conda run -p "$DIAPER_ENV" --no-capture-output python diaper/infer.py \
        -c "$cfg" \
        --epochs "$epochs_range" \
        --num-threads 4

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished inference: $name"

    median_window_length=$(grep '^median_window_length:' "$cfg" | sed 's/^median_window_length: *//')
    # Scoped to this run's own epochs<range>/ dir (see infer.py's out_dir:
    # rttms_dir/epochs<range>/timeshuffle.../spk_qty..._spk_qty_thr.../
    # detection_thr.../median<N>/subsampling.../rttms/*.rttm) -- an
    # unscoped `find "$rttms_dir" -path "*/median.../*.rttm"` would also
    # match any epochs<other-range>/ left over from an earlier run against
    # the same rttms_dir (e.g. before more checkpoints landed and widened
    # the auto-detected range), silently scoring against stale RTTMs.
    mapfile -t sys_rttms < <(find "$rttms_dir/epochs${epochs_range}" \
        -path "*/median${median_window_length}/*/rttms/*.rttm" -type f)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        echo "  WARNING: no .rttm files found under $rttms_dir/epochs${epochs_range} (median${median_window_length}), skipping scoring"
        continue
    fi

    collar_args=()
    collar_label="0"
    if [ -n "$collar" ]; then
        collar_args=(--collar "$collar")
        collar_label="$collar"
    fi

    score_log="${rttms_dir}/dscore_collar${collar_label}.log"
    echo "  scoring ${#sys_rttms[@]} RTTM(s) against $ref_rttm (collar ${collar_label}s)"

    conda run -p "$DSCORE_ENV" --no-capture-output python -u "$DSCORE_SRC/score.py" \
        "${collar_args[@]}" \
        -r "$ref_rttm" \
        -s "${sys_rttms[@]}" \
        > "$score_log" 2>&1

    echo "  full dscore output: $score_log"
    grep -h "OVERALL" "$score_log" || echo "  WARNING: no OVERALL line found in $score_log -- check it for errors"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished scoring: $name"
done

echo "All runs complete."
