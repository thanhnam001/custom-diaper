#!/bin/bash
set -e

# Re-run inference for every MSDWild-finetuned model after the
# median_window_length/subsampling post-processing fix (all infer_msdwild*.yaml
# configs previously carried RAMC's 0s-collar values -- median_window_length: 1,
# subsampling: 5 -- instead of MSDWild's own 0.25s-collar values --
# median_window_length: 11, subsampling: 10 -- see the header comment of each
# config for the full rationale).
#
# Every infer_msdwild*.yaml still has server-only absolute paths baked in
# (infer_data_dir/models_path/rttms_dir), since those configs were written
# against the training server's filesystem. This script overrides all three
# on the command line to point at this machine's local copies instead:
#   - infer_data_dir  -> database/msdwild/kaldi/test (see CLAUDE.md's
#     "Local data layout")
#   - models_path/rttms_dir -> the config's own server path with the
#     /data/ocr/namvt17/custom-diaper/ prefix swapped for this repo root,
#     which is exactly how the mirrored experiments/ tree is laid out
#     locally (see CLAUDE.md's "Environment" section for the conda env used
#     below).
#
# Run from the repo root:
#   ./scripts/rerun_infer_msdwild.sh
#
# After each config's RTTMs are produced, this script also scores them
# against ground truth with dscore using MSDWild's 0.25s collar (see
# CLAUDE.md's "Evaluation" section). score.py prints a large per-file table
# to stdout, so full output is redirected to a log file per config and only
# the "*** OVERALL ***" line is echoed to the terminal -- check the log file
# for the per-recording breakdown.

# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/rerun_infer_msdwild.sh`), falling back to
# GPU 0 only if the caller didn't set one -- same convention as
# run_pipeline_conformer_kernel3_6blocks.sh. Each infer_msdwild*.yaml's
# `gpu: 1` only means "use CUDA" (infer.py:411 checks `>= 1`, not a device
# index), so this env var is what actually picks the physical GPU; `conda
# run` inherits it since it's exported here before that subprocess starts.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SERVER_PREFIX="/data/ocr/namvt17/custom-diaper/"
SERVER_INFER_DATA_DIR="/data/ocr/namvt17/dataset/diarization/msdwild/kaldi/test"
REF_RTTM="/data/ocr/namvt17/dataset/diarization/msdwild/rttms/few.val.rttm"
DIAPER_ENV="/data/ocr/namvt17/custom-diaper/.venv"
DSCORE_SRC="/data/ocr/namvt17/custom-diaper/dscore"
DSCORE_ENV="/data/ocr/namvt17/custom-diaper/dscore/.dscore"
MSDWILD_COLLAR="0.25"

# Configs with checkpoints available locally under experiments/ (epochs
# already set in the yaml -- see each file's header comment for how that
# range was picked). kernel3_6blocks and linear_nodiversity are skipped:
# no local checkpoints exist for those yet (their MSDWild finetune either
# hasn't been run, or its output hasn't been synced from the server).
CONFIGS=(
    "models/10attractors/SC_LibriSpeech_2spk_adapted1-10/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31/infer_msdwild_l2aentropy0.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31/infer_msdwild_mlp_unmaskeddiv.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31/infer_msdwild_mlp_unmaskeddiv_mixed.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel5/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_conformer_layernorm/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_linear_diversity/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_adapted1-10_conformer/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_adapted1-10_conformer_maskeddiversity/infer_msdwild.yaml"
)
SKIPPED=(
    "models/10attractors/SC_LibriSpeech_2spk_conformer_kernel3_6blocks/infer_msdwild.yaml"
    "models/10attractors/SC_LibriSpeech_2spk_linear_nodiversity/infer_msdwild.yaml"
)

for cfg in "${SKIPPED[@]}"; do
    echo "SKIPPING $cfg -- no local checkpoints under experiments/ yet"
done

for cfg in "${CONFIGS[@]}"; do
    server_models_path=$(grep "^models_path:" "$cfg" | sed 's/^models_path: *//')
    server_rttms_dir=$(grep "^rttms_dir:" "$cfg" | sed 's/^rttms_dir: *//')
    local_models_path="${server_models_path#$SERVER_PREFIX}"
    local_rttms_dir="${server_rttms_dir#$SERVER_PREFIX}"

    if [ ! -d "$local_models_path" ]; then
        echo "SKIPPING $cfg -- expected checkpoints not found at $local_models_path"
        continue
    fi

    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] running: $cfg"
    echo "  models_path: $local_models_path"
    echo "  rttms_dir:   $local_rttms_dir"
    echo "=================================================================="

    conda run -p "$DIAPER_ENV" --no-capture-output python diaper/infer.py \
        -c "$cfg" \
        --infer-data-dir "$SERVER_INFER_DATA_DIR" \
        --models-path "$local_models_path" \
        --rttms-dir "$local_rttms_dir" \
        --num-threads 4

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished inference: $cfg"

    mapfile -t sys_rttms < <(find "$local_rttms_dir" -path '*/median11/*.rttm' -type f)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        echo "  WARNING: no .rttm files found under $local_rttms_dir, skipping scoring"
        continue
    fi

    score_log="${local_rttms_dir}/dscore_collar${MSDWILD_COLLAR}.log"
    echo "  scoring ${#sys_rttms[@]} RTTM(s) against $REF_RTTM (collar ${MSDWILD_COLLAR}s)"

    conda run -p "$DSCORE_ENV" --no-capture-output python -u "$DSCORE_SRC/score.py" \
        --collar "$MSDWILD_COLLAR" \
        -r "$REF_RTTM" \
        -s "${sys_rttms[@]}" \
        > "$score_log" 2>&1

    echo "  full dscore output: $score_log"
    grep -h "OVERALL" "$score_log" || echo "  WARNING: no OVERALL line found in $score_log -- check it for errors"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished scoring: $cfg"
done

echo "All runs complete."
