#!/bin/bash
set -e

# Inference + dscore scoring for the subsampling-5 E-Branchformer(mlp)
# RAMC finetune (models/10attractors/
# SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/), swept
# over median_window_length -- see run_infer_2500h_paperlr_ebranchformer_subsampling5.sh
# (the MSDWild sibling) for the general reasoning. RAMC's own standing
# convention is median1/collar0 (Section IV.D), so median1 is expected to
# be competitive, but do not assume it's exactly optimal: a quick collar-0
# rescore on a MSDWild subset in this same investigation found median7 beat
# median1 in that setting -- untested for RAMC's own data, hence the sweep
# rather than a hardcoded median1-only run.
#
# Uses --collar 0 for dscore (RAMC's convention, NOT MSDWild's 0.25 --
# see CLAUDE.md's "Evaluation" section).
#
# Compare the result here against:
#   - the subsampling-10-trained sibling (models/10attractors/
#     SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/infer_ramc.yaml),
#     which is ALSO inferred at subsampling 5 despite being trained at 10
#     (RAMC's standing convention) -- so this comparison isolates whether
#     finetuning at the matched resolution helps, given both configs infer
#     at the same subsampling.
#   - the plain self-attention subsampling5 RAMC sibling, if/when it's
#     built (mirroring run_infer_2500h_paperlr_subsampling5.sh's relationship
#     to run_infer_2500h_paperlr_ebranchformer_subsampling5.sh).
#
# HARDWARE WARNING: num_frames: -1 (whole-recording inference) at
# subsampling 5 needs ~22GB+ for a single long RAMC recording -- run this
# on the server only, never on the local 6GB-GPU machine (0/43 RAMC test
# files fit there; see infer_ramc.yaml's header in this lineage's
# subsampling5 directory and [[diaper-ramc-infer-hardware-limit]]).
#
# Follows the server-path/conda-env conventions of
# run_infer_2500h_paperlr_ebranchformer.sh.
#
# Run from the repo root, on the server, after
# run_finetune_2500h_paperlr_ebranchformer_subsampling5_ramc.sh has
# produced checkpoints:
#   ./scripts/run_infer_2500h_paperlr_ebranchformer_subsampling5_ramc.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAPER_ENV="/data/ocr/namvt17/custom-diaper/.venv"
DSCORE_SRC="/data/ocr/namvt17/custom-diaper/dscore"
DSCORE_ENV="/data/ocr/namvt17/custom-diaper/dscore/.dscore"

MAX_CHECKPOINTS_TO_AVERAGE=10
MEDIAN_SWEEP=(1 3 5 7 11 15 21)

CFG="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/infer_ramc.yaml"

models_path=$(grep "^models_path:" "$CFG" | sed 's/^models_path: *//')
rttms_dir=$(grep "^rttms_dir:" "$CFG" | sed 's/^rttms_dir: *//')
infer_data_dir=$(grep "^infer_data_dir:" "$CFG" | sed 's/^infer_data_dir: *//')
ref_rttm="${infer_data_dir}/rttm"

if [ ! -d "$models_path" ]; then
    echo "ABORT -- no checkpoints found at $models_path"
    echo "  Run ./scripts/run_finetune_2500h_paperlr_ebranchformer_subsampling5_ramc.sh first."
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] subsampling-5 E-Branchformer(mlp) RAMC finetune"
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

    score_log="${rttms_dir}/dscore_collar0_median${MED}.log"
    conda run -p "$DSCORE_ENV" --no-capture-output python -u "$DSCORE_SRC/score.py" \
        --collar 0 \
        -r "$ref_rttm" \
        -s "${sys_rttms[@]}" \
        > "$score_log" 2>&1

    ov=$(grep -h "OVERALL" "$score_log" | awk '{print $4}')
    DER_BY_MEDIAN[$MED]="$ov"
    echo "  median${MED}: n=${#sys_rttms[@]}  DER=${ov:-<parse failed, check $score_log>}"
done

echo "=================================================================="
echo "SUMMARY (E-Branchformer(mlp), RAMC, subsampling 5, epochs ${epochs_range})"
echo "  reference: subsampling-10-trained sibling (inferred at subsampling5 per RAMC convention) -- see results.csv for its DER"
echo "  reference: paper's published RAMC-with-FT DER of 21.1"
for MED in "${MEDIAN_SWEEP[@]}"; do
    echo "  median${MED}: DER=${DER_BY_MEDIAN[$MED]:-FAILED}"
done
echo "=================================================================="
