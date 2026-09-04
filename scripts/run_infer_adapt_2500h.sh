#!/bin/bash
# Run inference (+ dscore scoring) for the ADAPT-STAGE checkpoint of every
# 2500h lineage, on both MSDWild and RAMC test.
#
# WHY THIS EXISTS
# Every 2500h lineage has been scored after in-domain finetuning, and none
# of them (bar three one-off MSDWild runs kept under results/
# adapt_stage_msdwild_compare/) has been scored at the adapt stage -- the
# point right after multi-speaker adaptation on synthetic data, before any
# MSDWild or RAMC finetuning. That is the column that tells you how much of
# each architecture's final DER was already there before the finetune
# stage, which is exactly the question left open by the finetune-stage
# analysis (the adapt-stage gap to the paper is ~1.4 DER, the finetuned gap
# ~3.2, so the finetune stage is where the lineages diverge). Filling the
# column here makes that comparison per-architecture instead of per-lineage.
#
# WHAT IT RUNS
# 8 lineages x 2 datasets = 16 runs. Each reuses the lineage's own existing
# infer_{msdwild,ramc}*.yaml -- so the architecture flags, the test data
# dir, and the per-dataset postprocessing (MSDWild: median 11 / subsampling
# 10 / collar 0.25; RAMC: median 1 / subsampling 5 / collar 0) all come from
# the config that already produced that lineage's finetuned numbers, and
# stay comparable to them. Only two things are overridden on the command
# line:
#
#   --models-path   <experiment>/models              (adapt, not finetuned)
#   --rttms-dir     <experiment>/<ds>_test_adapt_pred
#
# The experiment directory is DERIVED from the config's own models_path
# (which points at <experiment>/models_finetuneXXX/models), so this script
# hardcodes no experiment paths and follows the configs if the server root
# ever moves.
#
# Every adapt directory holds its last checkpoints at epochs 91..100, so
# all 16 runs average the same --epochs 90-100 window. The script still
# auto-detects rather than assuming, and prints what it picked.
#
# ON THE SERVER, from the repo root:
#
#   ./scripts/run_infer_adapt_2500h.sh                   # all 16, one GPU
#   DRY_RUN=1 ./scripts/run_infer_adapt_2500h.sh         # print the plan only
#   DATASETS=ramc ./scripts/run_infer_adapt_2500h.sh     # RAMC only (fast: 43 files)
#   ONLY='fixednoam.*' ./scripts/run_infer_adapt_2500h.sh
#
# ONLY is an ANCHORED extended regex over the lineage labels, not a
# substring -- ONLY=fixednoam matches nothing, ONLY='fixednoam.*' matches
# both. The eight labels are:
#
#   2500h_baseline  paperlr  paperlr_ebf  fixednoam_cnf
#   fixednoam_ebf   spkcounting  headoff  overlaploss3
#
# Two GPUs, two shells -- MSDWild is 490 files per lineage and RAMC only
# 43, so splitting by dataset is lopsided; splitting by lineage is not:
#
#   CUDA_VISIBLE_DEVICES=0 ONLY='2500h_baseline|paperlr|paperlr_ebf|fixednoam_cnf' \
#       ./scripts/run_infer_adapt_2500h.sh
#   CUDA_VISIBLE_DEVICES=1 ONLY='fixednoam_ebf|spkcounting|headoff|overlaploss3' \
#       ./scripts/run_infer_adapt_2500h.sh
#
# RESUMABLE. infer.py skips any recording whose RTTM already exists, and
# this script skips a run outright once its output directory holds one RTTM
# per line of the test set's wav.scp. Ctrl-C and re-run costs only the
# recording that was in flight.
#
# Weights come from scripts/pack_adapt_weights_for_server.sh.

set -eu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"

MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"
NUM_THREADS="${NUM_THREADS:-4}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-msdwild ramc}"
ONLY="${ONLY:-}"
CFG_ROOT="${CFG_ROOT:-models/10attractors}"

# label|config dir|msdwild config|ramc config
#
# The two config-file naming conventions in this repo are both represented:
# the _spkcounting* lineages use infer_*_mlp.yaml, the rest infer_*.yaml.
LINEAGES=(
    "2500h_baseline|SC_LibriSpeech_2spk_2500h|infer_msdwild.yaml|infer_ramc.yaml"
    "paperlr|SC_LibriSpeech_2spk_2500h_paperlr|infer_msdwild.yaml|infer_ramc.yaml"
    "paperlr_ebf|SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer|infer_msdwild.yaml|infer_ramc.yaml"
    "fixednoam_cnf|SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31|infer_msdwild.yaml|infer_ramc.yaml"
    "fixednoam_ebf|SC_LibriSpeech_2spk_2500h_fixednoam_ebf|infer_msdwild.yaml|infer_ramc.yaml"
    "spkcounting|SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting|infer_msdwild_mlp.yaml|infer_ramc_mlp.yaml"
    "headoff|SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff|infer_msdwild_mlp.yaml|infer_ramc_mlp.yaml"
    "overlaploss3|SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_overlaploss3|infer_msdwild_mlp.yaml|infer_ramc_mlp.yaml"
)

# Same helper as run_5day_queue.sh / pack_results_from_server.sh, on
# purpose: if those resolve a path, so does this one.
yaml_get () {  # yaml_get <key> <file>
    grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"
}

ran=0; skipped=0; failed=0
declare -a SUMMARY=()

for lineage in "${LINEAGES[@]}"; do
    IFS='|' read -r label cfgdir msd_cfg ramc_cfg <<< "$lineage"

    if [ -n "$ONLY" ] && ! echo "$label" | grep -Eq "^($ONLY)$"; then
        continue
    fi

    for ds in $DATASETS; do
        case "$ds" in
            msdwild) cfg="$CFG_ROOT/$cfgdir/$msd_cfg";  collar="0.25" ;;
            ramc)    cfg="$CFG_ROOT/$cfgdir/$ramc_cfg"; collar=""     ;;
            *) echo "ERROR: unknown dataset '$ds' (want msdwild or ramc)" >&2; exit 1 ;;
        esac
        name="${label}_adapt_${ds}"

        if [ ! -f "$cfg" ]; then
            echo "SKIPPING $name -- no such config: $cfg"
            skipped=$(( skipped + 1 ))
            continue
        fi

        # The config points at the FINETUNED weights; walk two levels up to
        # the experiment dir, then back down to the adapt-stage models/.
        ft_models_path=$(yaml_get models_path "$cfg")
        expdir=$(dirname "$(dirname "$ft_models_path")")
        models_path="$expdir/models"
        rttms_dir="$expdir/${ds}_test_adapt_pred"
        infer_data_dir=$(yaml_get infer_data_dir "$cfg")
        ref_rttm="${infer_data_dir}/rttm"

        # Guard against a config whose models_path is not <exp>/models_finetuneX/models
        # -- without this, a mis-shaped path would silently resolve to some
        # other directory's checkpoints and score them under this label.
        case "$(basename "$expdir")" in
            *adapted1-10*) ;;
            *)
                echo "SKIPPING $name -- '$ft_models_path' in $cfg does not look like"
                echo "  <experiment>/models_finetuneXXX/models (got expdir '$expdir')"
                skipped=$(( skipped + 1 ))
                continue
                ;;
        esac

        if [ ! -d "$models_path" ]; then
            echo "SKIPPING $name -- no adapt checkpoints at $models_path"
            echo "  (unpack diaper_adapt_weights.tar first -- see scripts/pack_adapt_weights_for_server.sh)"
            skipped=$(( skipped + 1 ))
            continue
        fi

        mapfile -t ckpt_epochs < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
            -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
        if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
            echo "SKIPPING $name -- $models_path exists but has no checkpoint_*.tar files"
            skipped=$(( skipped + 1 ))
            continue
        fi

        last_idx=$(( ${#ckpt_epochs[@]} - 1 ))
        last_epoch="${ckpt_epochs[$last_idx]}"
        if [ "${#ckpt_epochs[@]}" -gt "$MAX_CHECKPOINTS_TO_AVERAGE" ]; then
            start_idx=$(( ${#ckpt_epochs[@]} - MAX_CHECKPOINTS_TO_AVERAGE ))
        else
            start_idx=0
        fi
        first_epoch_averaged="${ckpt_epochs[$start_idx]}"
        # parse_epochs' "-" syntax averages (start, end] -- subtract 1 so the
        # first epoch in ckpt_epochs is actually included.
        epochs_range="$(( first_epoch_averaged - 1 ))-${last_epoch}"

        median_window_length=$(yaml_get median_window_length "$cfg")
        subsampling=$(yaml_get subsampling "$cfg")

        # How many recordings this test set has -- one RTTM per wav.scp line
        # when the run is complete.
        expected=0
        if [ -f "$infer_data_dir/wav.scp" ]; then
            expected=$(wc -l < "$infer_data_dir/wav.scp")
        fi
        got=$(find "$rttms_dir/epochs${epochs_range}" -name '*.rttm' -type f 2>/dev/null | wc -l)

        echo "=================================================================="
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name"
        echo "  config:      $cfg"
        echo "  data:        $infer_data_dir  ($expected recordings)"
        echo "  adapt model: $models_path"
        echo "  out:         $rttms_dir"
        echo "  epochs:      $epochs_range (averaging last $(( last_idx - start_idx + 1 )) of ${#ckpt_epochs[@]} checkpoint(s))"
        echo "  postproc:    median${median_window_length} subsampling${subsampling}, dscore collar ${collar:-0}"
        echo "=================================================================="

        if [ "$DRY_RUN" = "1" ]; then
            echo "  DRY_RUN=1 -- not running ($got/$expected RTTMs present)"
            continue
        fi

        if [ "$expected" -gt 0 ] && [ "$got" -ge "$expected" ]; then
            echo "  already complete ($got/$expected RTTMs) -- skipping inference"
        else
            if ! conda run -p "$DIAPER_ENV" --no-capture-output python diaper/infer.py \
                -c "$cfg" \
                --models-path "$models_path" \
                --rttms-dir "$rttms_dir" \
                --epochs "$epochs_range" \
                --num-threads "$NUM_THREADS"; then
                echo "  ERROR: inference failed for $name -- continuing with the next run"
                SUMMARY+=("FAIL   $name  (inference)")
                failed=$(( failed + 1 ))
                continue
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished inference: $name"
        fi

        # Scoped to this run's own epochs<range>/ dir (infer.py writes
        # rttms_dir/epochs<range>/timeshuffle.../spk_qty..._spk_qty_thr.../
        # detection_thr.../median<N>/subsampling.../rttms/*.rttm) -- an
        # unscoped find would also match an epochs<other-range>/ left over
        # from an earlier run against the same rttms_dir, silently scoring
        # stale RTTMs.
        mapfile -t sys_rttms < <(find "$rttms_dir/epochs${epochs_range}" \
            -path "*/median${median_window_length}/*/rttms/*.rttm" -type f)
        if [ "${#sys_rttms[@]}" -eq 0 ]; then
            echo "  WARNING: no .rttm under $rttms_dir/epochs${epochs_range} (median${median_window_length}), skipping scoring"
            SUMMARY+=("FAIL   $name  (no RTTMs)")
            failed=$(( failed + 1 ))
            continue
        fi
        if [ "$expected" -gt 0 ] && [ "${#sys_rttms[@]}" -ne "$expected" ]; then
            echo "  WARNING: scoring ${#sys_rttms[@]} RTTM(s) but the test set has $expected recordings"
            echo "  -- DER below is over a SUBSET and is not comparable to the finetuned numbers."
        fi

        collar_args=()
        collar_label="0"
        if [ -n "$collar" ]; then
            collar_args=(--collar "$collar")
            collar_label="$collar"
        fi

        score_log="${rttms_dir}/dscore_collar${collar_label}.log"
        echo "  scoring ${#sys_rttms[@]} RTTM(s) against $ref_rttm (collar ${collar_label}s)"

        if ! conda run -p "$DSCORE_ENV" --no-capture-output python -u "$DSCORE_SRC/score.py" \
            "${collar_args[@]}" \
            -r "$ref_rttm" \
            -s "${sys_rttms[@]}" \
            > "$score_log" 2>&1; then
            echo "  ERROR: dscore failed -- see $score_log"
            SUMMARY+=("FAIL   $name  (dscore)")
            failed=$(( failed + 1 ))
            continue
        fi

        echo "  full dscore output: $score_log"
        overall=$(grep -h "OVERALL" "$score_log" || true)
        if [ -z "$overall" ]; then
            echo "  WARNING: no OVERALL line in $score_log -- check it for errors"
            SUMMARY+=("FAIL   $name  (no OVERALL line)")
            failed=$(( failed + 1 ))
            continue
        fi
        echo "$overall"
        # dscore's OVERALL row is whitespace-separated with DER in column 4.
        der=$(echo "$overall" | awk '{print $4}')
        SUMMARY+=("$(printf 'OK     %-28s DER %-8s (%d files, collar %s)' "$name" "$der" "${#sys_rttms[@]}" "$collar_label")")
        ran=$(( ran + 1 ))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished scoring: $name"
    done
done

echo
echo "=================================================================="
echo "ADAPT-STAGE SUMMARY  ($ran scored, $skipped skipped, $failed failed)"
echo "=================================================================="
for line in "${SUMMARY[@]:-}"; do
    [ -n "$line" ] && echo "  $line"
done
echo
echo "Compare each row against the same lineage's FINETUNED DER to see how"
echo "much of it the finetune stage actually bought."
