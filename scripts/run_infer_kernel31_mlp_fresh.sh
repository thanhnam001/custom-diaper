#!/bin/bash
set -e

# Runs inference (+ dscore scoring) for all 4 finetuned checkpoints of the
# SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh lineage
# (models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh/):
#   - MSDWild finetune, pipeline 2 (mlp only):        infer_msdwild_mlp.yaml
#   - MSDWild finetune, pipeline 1 (mlp+unmaskeddiv):  infer_msdwild_mlp_unmaskeddiv.yaml
#   - RAMC finetune,    pipeline 2 (mlp only):         infer_ramc_mlp.yaml
#   - RAMC finetune,    pipeline 1 (mlp+unmaskeddiv):  infer_ramc_mlp_unmaskeddiv.yaml
#
# Those 4 configs carry server-only absolute paths (infer_data_dir/
# models_path/rttms_dir, written against the training server's filesystem)
# and no `epochs` (no checkpoints existed locally when they were created --
# see each config's header comment). This script overrides all of that on
# the command line:
#   - infer_data_dir  -> database/msdwild|ramc/kaldi/test (this machine's
#     local copy, see CLAUDE.md's "Local data layout")
#   - models_path/rttms_dir -> the config's own server path with the
#     /data/ocr/namvt17/custom-diaper/ prefix swapped for this repo root,
#     matching how the mirrored experiments/ tree is laid out locally (same
#     convention as scripts/rerun_infer_msdwild.sh)
#   - epochs -> auto-detected from whatever checkpoint_<N>.tar files exist
#     under the local models_path, averaging the last
#     $MAX_CHECKPOINTS_TO_AVERAGE of them ("N-1 to M" parse_epochs syntax,
#     see diaper/backend/models.py::parse_epochs). There's no dev-DER curve
#     to hand-pick a range from yet, so this is a reasonable default -- redo
#     the run with an explicit `--epochs` once TensorBoard shows a better
#     range.
#
# A config is skipped if its local models_path has no checkpoints yet
# (nothing synced from the server for that pipeline/dataset combination).
#
# Uses this machine's local diaper/dscore envs and checkouts per CLAUDE.md's
# "Environment" / "Evaluation" sections (NOT the server-only env paths
# scripts/rerun_infer_msdwild.sh hardcodes).
#
# Run from the repo root:
#   ./scripts/run_infer_kernel31_mlp_fresh.sh

# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell,
# falling back to GPU 0 only if the caller didn't set one -- same
# convention as the run_pipeline_*.sh scripts. Each infer_*.yaml's `gpu: 1`
# only means "use CUDA" (infer.py checks `>= 1`, not a device index), so
# this env var is what actually picks the physical GPU; `conda run`
# inherits it since it's exported here before that subprocess starts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAPER_ENV="../Master/repos/DiaPer/.diaper_env"
DSCORE_ENV_NAME="dscore"
DSCORE_SRC="../Master/repos/dscore"

SERVER_PREFIX="/data/ocr/namvt17/custom-diaper/"
MAX_CHECKPOINTS_TO_AVERAGE=10

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh"

# name|config|local infer_data_dir|reference rttm|dscore collar (empty = no --collar flag, i.e. 0s)
RUNS=(
    "msdwild_mlp|${CONFIG_DIR}/infer_msdwild_mlp.yaml|database/msdwild/kaldi/test|database/msdwild/kaldi/test/rttm|0.25"
    "msdwild_mlp_unmaskeddiv|${CONFIG_DIR}/infer_msdwild_mlp_unmaskeddiv.yaml|database/msdwild/kaldi/test|database/msdwild/kaldi/test/rttm|0.25"
    "ramc_mlp|${CONFIG_DIR}/infer_ramc_mlp.yaml|database/ramc/kaldi/test|database/ramc/kaldi/test/rttm|"
    "ramc_mlp_unmaskeddiv|${CONFIG_DIR}/infer_ramc_mlp_unmaskeddiv.yaml|database/ramc/kaldi/test|database/ramc/kaldi/test/rttm|"
)

for run in "${RUNS[@]}"; do
    IFS='|' read -r name cfg local_data ref_rttm collar <<< "$run"

    server_models_path=$(grep "^models_path:" "$cfg" | sed 's/^models_path: *//')
    server_rttms_dir=$(grep "^rttms_dir:" "$cfg" | sed 's/^rttms_dir: *//')
    local_models_path="${server_models_path#$SERVER_PREFIX}"
    local_rttms_dir="${server_rttms_dir#$SERVER_PREFIX}"

    if [ ! -d "$local_models_path" ]; then
        echo "SKIPPING $name ($cfg) -- no local checkpoints found at $local_models_path"
        continue
    fi

    mapfile -t ckpt_epochs < <(find "$local_models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
        -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
        echo "SKIPPING $name ($cfg) -- $local_models_path exists but has no checkpoint_*.tar files"
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
    echo "  models_path: $local_models_path"
    echo "  rttms_dir:   $local_rttms_dir"
    echo "  epochs:      $epochs_range (averaging ${#ckpt_epochs[@]} available checkpoint(s), capped at $MAX_CHECKPOINTS_TO_AVERAGE)"
    echo "=================================================================="

    conda run -p "$DIAPER_ENV" --no-capture-output python diaper/infer.py \
        -c "$cfg" \
        --infer-data-dir "$local_data" \
        --models-path "$local_models_path" \
        --rttms-dir "$local_rttms_dir" \
        --epochs "$epochs_range" \
        --num-threads 4

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished inference: $name"

    median_window_length=$(grep '^median_window_length:' "$cfg" | sed 's/^median_window_length: *//')
    mapfile -t sys_rttms < <(find "$local_rttms_dir" -path "*/median${median_window_length}/*.rttm" -type f)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        echo "  WARNING: no .rttm files found under $local_rttms_dir (median${median_window_length}), skipping scoring"
        continue
    fi

    collar_args=()
    collar_label="0"
    if [ -n "$collar" ]; then
        collar_args=(--collar "$collar")
        collar_label="$collar"
    fi

    score_log="${local_rttms_dir}/dscore_collar${collar_label}.log"
    echo "  scoring ${#sys_rttms[@]} RTTM(s) against $ref_rttm (collar ${collar_label}s)"

    conda run -n "$DSCORE_ENV_NAME" --no-capture-output python -u "$DSCORE_SRC/score.py" \
        "${collar_args[@]}" \
        -r "$ref_rttm" \
        -s "${sys_rttms[@]}" \
        > "$score_log" 2>&1

    echo "  full dscore output: $score_log"
    grep -h "OVERALL" "$score_log" || echo "  WARNING: no OVERALL line found in $score_log -- check it for errors"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished scoring: $name"
done

echo "All runs complete."
