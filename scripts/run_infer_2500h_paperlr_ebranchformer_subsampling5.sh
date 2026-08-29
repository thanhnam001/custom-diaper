#!/bin/bash
set -e

# Inference + dscore scoring for the subsampling-5 E-Branchformer(mlp)
# MSDWild finetune (models/10attractors/
# SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/), swept
# over median_window_length -- see run_infer_2500h_paperlr_subsampling5.sh
# for the full reasoning (identical here, different architecture).
#
# Compare the best median setting found here against:
#   - the subsampling-10 sibling (models/10attractors/
#     SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/infer_msdwild.yaml),
#     MSDWild 18.61 -- same init checkpoint, only subsampling/num_frames
#     changed
#   - the plain self-attention subsampling5 sibling's result
#     (scripts/run_infer_2500h_paperlr_subsampling5.sh) -- self-attention
#     has no fixed-kernel handicap at the finer resolution, E-Branchformer
#     does (kernel_size left at 31 taps = 1.55s receptive field here vs
#     3.1s at subsampling 10, see this lineage's finetune config header).
#     If self-attention improves more than E-Branchformer from this
#     change, that is evidence the halved receptive field is costing
#     E-Branchformer here, not that resolution doesn't help conv encoders
#     in general.
#
# Follows the server-path/conda-env conventions of
# run_infer_2500h_paperlr_ebranchformer.sh.
#
# Run from the repo root, on the server, after
# run_finetune_2500h_paperlr_ebranchformer_subsampling5.sh has produced
# checkpoints:
#   ./scripts/run_infer_2500h_paperlr_ebranchformer_subsampling5.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAPER_ENV="/data/ocr/namvt17/custom-diaper/.venv"
DSCORE_SRC="/data/ocr/namvt17/custom-diaper/dscore"
DSCORE_ENV="/data/ocr/namvt17/custom-diaper/dscore/.dscore"

MAX_CHECKPOINTS_TO_AVERAGE=10
MEDIAN_SWEEP=(1 3 5 7 11 15 21)

CFG="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/infer_msdwild.yaml"

models_path=$(grep "^models_path:" "$CFG" | sed 's/^models_path: *//')
rttms_dir=$(grep "^rttms_dir:" "$CFG" | sed 's/^rttms_dir: *//')
infer_data_dir=$(grep "^infer_data_dir:" "$CFG" | sed 's/^infer_data_dir: *//')
ref_rttm="${infer_data_dir}/rttm"

if [ ! -d "$models_path" ]; then
    echo "ABORT -- no checkpoints found at $models_path"
    echo "  Run ./scripts/run_finetune_2500h_paperlr_ebranchformer_subsampling5.sh first."
    exit 1
fi

mapfile -t ckpt_epochs < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
    -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
    echo "ABORT -- $models_path exists but has no checkpoint_*.tar files"
    exit 1
fi

last_idx=$(( ${#ckpt_epochs[@]} - 1 ))
last_epoch="${ckpt_epochs[$last_idx]}"
start_idx=$(( ${#ckpt_epochs[@]} > MAX_CHECKPOINTS_TO_AVERAGE ? ${#ckpt_epochs[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
first_epoch_averaged="${ckpt_epochs[$start_idx]}"
epochs_range="$(( first_epoch_averaged - 1 ))-${last_epoch}"

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] subsampling-5 E-Branchformer(mlp) MSDWild finetune"
echo "  models_path: $models_path"
echo "  epochs:      $epochs_range (averaging ${#ckpt_epochs[@]} checkpoint(s), capped at $MAX_CHECKPOINTS_TO_AVERAGE)"
echo "  median sweep: ${MEDIAN_SWEEP[*]}"
echo "=================================================================="

declare -A DER_BY_MEDIAN

for MED in "${MEDIAN_SWEEP[@]}"; do
    echo "-- median${MED} --"
    conda run -p "$DIAPER_ENV" --no-capture-output python diaper/infer.py \
        -c "$CFG" \
        --epochs "$epochs_range" \
        --median-window-length "$MED" \
        --num-threads 4

    sys_dir="${rttms_dir}/epochs${epochs_range}"
    mapfile -t sys_rttms < <(find "$sys_dir" \
        -path "*/median${MED}/*/rttms/*.rttm" -type f)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        echo "  WARNING: no .rttm files found for median${MED}, skipping scoring"
        continue
    fi

    score_log="${rttms_dir}/dscore_collar0.25_median${MED}.log"
    conda run -p "$DSCORE_ENV" --no-capture-output python -u "$DSCORE_SRC/score.py" \
        --collar 0.25 \
        -r "$ref_rttm" \
        -s "${sys_rttms[@]}" \
        > "$score_log" 2>&1

    ov=$(grep -h "OVERALL" "$score_log" | awk '{print $4}')
    DER_BY_MEDIAN[$MED]="$ov"
    echo "  median${MED}: n=${#sys_rttms[@]}  DER=${ov:-<parse failed, check $score_log>}"
done

echo "=================================================================="
echo "SUMMARY (E-Branchformer(mlp), subsampling 5, epochs ${epochs_range})"
echo "  reference: subsampling-10 sibling scored 18.61 at median11"
for MED in "${MEDIAN_SWEEP[@]}"; do
    echo "  median${MED}: DER=${DER_BY_MEDIAN[$MED]:-FAILED}"
done
echo "=================================================================="
