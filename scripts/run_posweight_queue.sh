#!/bin/bash
# 2-GPU queue: fresh MSDWild finetunes of the E-Branchformer(mlp) paperlr
# lineage with a rebalanced attractor-existence BCE.
#
#   mkdir -p logs/posweight_queue
#   nohup ./scripts/run_posweight_queue.sh > logs/posweight_queue/driver.log 2>&1 &
#
#
# WHAT THIS TESTS AND WHY
# ------------------------
# losses.py::pit_loss_multispk computes attractor_existence_loss as an
# UNWEIGHTED BCE over n_attractors (10) slots, while a MSDWild recording
# actually has 2-4 speakers -- so ~70% of the targets are "slot empty" and
# the head is trained biased toward absent. infer.py thresholds that same
# head directly (infer.py:238) to decide how many speakers to emit.
#
# Measured consequence (per-file md-eval over 490 test files, 2026-09-10):
# our best checkpoint under-predicts speaker count on 38.0% of files
# (mean n_sys - n_ref = -0.40) vs the paper's own checkpoint at 32.4% /
# -0.30, and the residual DER gap tracks exactly that.
#
# The inference-side fixes are EXHAUSTED, both directions:
#   - existence-gating threshold 0.5 -> 0.3 -> 0.1: bit-for-bit no change
#     (the head is saturated-confident, not marginally wrong).
#   - frame-activation threshold swept DOWN (msdwild_der_gap_analysis.md)
#     and UP (2026-09-10, on the sub5 model): 0.5 is already near-optimal.
# So the only remaining handle is training-side, which is what this runs:
# --attractor-existence-pos-weight upweights present-speaker slots INSIDE
# that BCE. With 10 slots and ~2.5 speakers the pos:neg ratio is ~1:3, so
# 3.0 is the principled balance point; 5.0 is a deliberate overshoot that
# tells us whether the lever has traction at all even if 3.0's effect is
# small. If neither moves under-prediction off ~38%, the lever is dead.
#
# NOT the same knob as the config's attractor_existence_loss_weight (1.0),
# which scales the WHOLE existence term against the activation/diversity
# losses. Easy to grab the wrong one.
#
#
# WHY FRESH FINETUNE, NOT A WARM START
# -------------------------------------
# Warm-starting the converged ep750 checkpoint would be uninterpretable:
# the existence head is already saturated (that is exactly what the
# zero-effect threshold sweep proves), so a reweighted loss at lr 1e-6
# would barely move it and a null result would not distinguish "pos_weight
# does not work" from "we never moved the head". Training from the adapt
# init, before saturation, is the only way this answers its own question.
#
#
# WHY train_batchsize STAYS 32 ON A 32 GB CARD
# ---------------------------------------------
# Yes, 32 GB fits more at subsampling 10 / num_frames 600 -- memory is not
# the binding constraint here, COMPARABILITY is. The control arm for this
# experiment is the existing models_finetuneMSDWILD run (pos_weight 1.0),
# trained at batch 32, and the whole design rests on comparing at MATCHED
# EPOCHS. The finetune stage uses flat `lr: 1e-6` with Adam (no Noam here
# -- that lives in the pretrain/adapt configs), so doubling the batch
# halves steps-per-epoch and therefore roughly halves optimization progress
# per epoch. The pos_weight arms would then look worse than the control for
# a reason that has nothing to do with pos_weight. This project has already
# been burned by exactly this: see memory diaper-fixednoam-conformer-queue,
# where a conformer arm's 23.71 -> 24.36 was "partly an epoch-budget
# artifact" at epoch 162 vs the baseline's 282.
#
# If you want to spend the spare VRAM, spend it on MORE ARMS at batch 32
# (a second process per GPU), not a bigger batch on fewer arms. Raising
# `dev_batchsize` is also free -- validation only, no training dynamics.
#
#
# ENV KNOBS
#   LANE_A_GPU / LANE_B_GPU   physical device ids   (default 0 / 1)
#   POSW_A / POSW_B           pos_weight per lane   (default 3.0 / 5.0)
#   ONLY_LANE=A|B             run a single lane
#   PATIENCE                  default 200 (config's 100 fired early before)
#   MAX_EPOCHS                default 750 (matches the control's budget)
#   SCORE_ONLY=1              skip training, just score what is on disk now
#                             -- safe to run in another shell mid-flight
#   SKIP_PREFLIGHT=1          don't verify the adapt checkpoints first
#   LOG_DIR                   default logs/posweight_queue

set -u

LANE_A_GPU="${LANE_A_GPU:-0}"
LANE_B_GPU="${LANE_B_GPU:-1}"
POSW_A="${POSW_A:-3.0}"
POSW_B="${POSW_B:-5.0}"
ONLY_LANE="${ONLY_LANE:-}"
PATIENCE="${PATIENCE:-200}"
MAX_EPOCHS="${MAX_EPOCHS:-750}"
SCORE_ONLY="${SCORE_ONLY:-0}"
LOG_DIR="${LOG_DIR:-logs/posweight_queue}"

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"
MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"

EBF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer
CFG="$EBF_DIR/finetune_msdwild_10spks.yaml"
INFER_CFG="$EBF_DIR/infer_msdwild.yaml"

mkdir -p "$LOG_DIR"
STOP_FILE="${LOG_DIR}/STOP"

if [ "${USE_CONDA_RUN:-1}" = "1" ]; then
    PY=(conda run -p "$DIAPER_ENV" --no-capture-output python)
    DSCORE_PY=(conda run -p "$DSCORE_ENV" --no-capture-output python -u)
else
    PY=(python)
    DSCORE_PY=(python -u)
fi

log () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
yaml_get () { grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"; }
stop_requested () { [ -f "$STOP_FILE" ]; }

# Same escalation as run_36h_resume_queue.sh: a background lane subshell
# ignores a bare SIGINT, so tear down the whole process group.
on_signal () {
    trap - INT TERM
    log "caught SIG$1 -- stopping the queue"
    : > "$STOP_FILE"
    local mypgid
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    if [ -n "$mypgid" ] && [ "$mypgid" = "$$" ]; then
        kill -TERM 0 2>/dev/null; sleep 5; kill -KILL 0 2>/dev/null
    else
        pkill -TERM -P $$ 2>/dev/null
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

BASELINE_OUT="$(yaml_get output_path "$CFG")"
ADAPT_INIT="$(yaml_get init_model_path "$CFG")"

# ---------------------------------------------------------------------------
# preflight: the adapt checkpoints must exist, and we must NOT be pointed at
# the control run's directory.
#
# Both of these have bitten this project before. The 5-day queue's lane-B D1
# produced no log at all because its models_path was missing on the server
# (memory diaper-fixednoam-conformer-queue). And train.py auto-resumes from
# the latest checkpoint in output_path BEFORE considering init_model_path --
# so if output_path were left at the config's value, this would silently
# become a warm start from the control's epoch 750 *and overwrite the
# control's checkpoints*. That would destroy the only baseline we have.
# ---------------------------------------------------------------------------
preflight () {
    local out="$1" label="$2" n
    if [ "$out" = "$BASELINE_OUT" ]; then
        log "FATAL $label -- output_path equals the control run ($BASELINE_OUT)."
        log "      That would overwrite the pos_weight=1.0 baseline. Refusing."
        return 1
    fi
    n=$(find "$ADAPT_INIT" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        log "FATAL $label -- no adapt checkpoints at $ADAPT_INIT"
        log "      A fresh finetune warm-starts from the adapt stage"
        log "      (init_epochs $(yaml_get init_epochs "$CFG")). Upload them with"
        log "      scripts/pack_adapt_weights_for_server.sh, or this trains from"
        log "      random init and answers a different question entirely."
        return 1
    fi
    log "  ok $label -- adapt init has $n checkpoint(s), output -> $out"
    return 0
}

# ---------------------------------------------------------------------------
# train_arm <label> <gpu> <pos_weight> <output_path> <logfile>
# Deliberately does NOT override --init-model-path (unlike the resume queue):
# a fresh finetune is supposed to warm-start from the adapt checkpoints.
# train.py checkpoints every epoch and auto-resumes from output_path, so
# re-running this script after an interruption is safe and cheap.
# ---------------------------------------------------------------------------
train_arm () {
    local label="$1" gpu="$2" posw="$3" out="$4" logfile="$5"
    if stop_requested; then log "STOP requested -- not starting $label"; return 1; fi
    log "START $label gpu=$gpu pos_weight=$posw patience=$PATIENCE max_epochs=$MAX_EPOCHS"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$CFG" \
        --gpu 1 \
        --attractor-existence-pos-weight "$posw" \
        --early-stopping-patience "$PATIENCE" \
        --max-epochs "$MAX_EPOCHS" \
        --output-path "$out" >> "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "DONE  $label (patience fired or max_epochs reached)"; return 0
    fi
    log "FAIL  $label (exit $rc) -- see $logfile"
    if stop_requested; then return 1; fi
    log "RETRY $label once (train.py resumes from its own checkpoints)"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$CFG" \
        --gpu 1 \
        --attractor-existence-pos-weight "$posw" \
        --early-stopping-patience "$PATIENCE" \
        --max-epochs "$MAX_EPOCHS" \
        --output-path "$out" >> "$logfile" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { log "DONE  $label (after retry)"; return 0; }
    log "FAIL  $label (exit $rc, after retry)"; return 1
}

# ---------------------------------------------------------------------------
# score_arm <label> <gpu> <output_path> <logfile>
# MSDWild scoring is GPU-and-inline (490 files, ~2 min) -- none of RAMC's
# CPU/RAM locking applies. Averages the last $MAX_CHECKPOINTS_TO_AVERAGE
# checkpoints, whatever epoch the arm actually reached.
# ---------------------------------------------------------------------------
score_arm () {
    local label="$1" gpu="$2" out="$3" logfile="$4"
    local models_path="$out/models" rttms_dir="$out/msdwild_test_pred"

    if [ ! -d "$models_path" ]; then
        log "SKIP score $label -- no models dir at $models_path"; return 1
    fi
    mapfile -t ck < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
        -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#ck[@]}" -eq 0 ]; then
        log "SKIP score $label -- no checkpoint_*.tar yet"; return 1
    fi
    local last_idx=$(( ${#ck[@]} - 1 )) start_idx first last range
    last="${ck[$last_idx]}"
    start_idx=$(( ${#ck[@]} > MAX_CHECKPOINTS_TO_AVERAGE \
                  ? ${#ck[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
    first="${ck[$start_idx]}"
    range="$(( first - 1 ))-${last}"

    local infer_data_dir ref_rttm
    infer_data_dir=$(yaml_get infer_data_dir "$INFER_CFG")
    ref_rttm="${infer_data_dir}/rttm"

    log "START score $label epochs=$range"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/infer.py -c "$INFER_CFG" \
        --models-path "$models_path" --rttms-dir "$rttms_dir" \
        --epochs "$range" --median-window-length 11 --subsampling 10 \
        --gpu 1 --num-threads 4 >> "$logfile" 2>&1
    if [ $? -ne 0 ]; then log "FAIL  score $label -- see $logfile"; return 1; fi

    mapfile -t sys_rttms < <(find "$rttms_dir/epochs${range}" \
        -path "*/median11/*/rttms/*.rttm" -type f 2>/dev/null)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        log "WARN  no RTTMs under $rttms_dir/epochs${range} -- not scored"; return 1
    fi
    local score_log="${rttms_dir}/dscore_collar0.25_epochs${range}.log"
    "${DSCORE_PY[@]}" "$DSCORE_SRC/score.py" --collar 0.25 \
        -r "$ref_rttm" -s "${sys_rttms[@]}" > "$score_log" 2>&1
    log "SCORED $label (${#sys_rttms[@]} files) -> $score_log"
    grep -h "OVERALL" "$score_log" || log "WARN no OVERALL line in $score_log"

    # Primary read-out for THIS experiment is speaker-count calibration, not
    # DER -- dscore does not report it, and it is far less noisy than pooled
    # MSDWild DER (which cannot resolve differences under ~1.2, see memory
    # diaper-der-statistical-power). Control to beat: 38.0% under / -0.40.
    # Paper's own checkpoint: 32.4% under / -0.30.
    "${PY[@]}" - "$ref_rttm" "$rttms_dir/epochs${range}" <<'PYEOF' 2>&1 | tee -a "$logfile"
import sys, glob, os
from collections import defaultdict

ref_fn, sysroot = sys.argv[1], sys.argv[2]

def spk_by_file(lines):
    d = defaultdict(set)
    for ln in lines:
        p = ln.split()
        if len(p) > 7 and p[0] == 'SPEAKER':
            d[p[1]].add(p[7])
    return d

with open(ref_fn) as f:
    ref = spk_by_file(f)

sys_lines = []
for fn in glob.glob(os.path.join(sysroot, '**', 'rttms', '*.rttm'), recursive=True):
    with open(fn) as f:
        sys_lines.extend(f)
hyp = spk_by_file(sys_lines)

common = [k for k in ref if k in hyp]
if not common:
    print('  spk-count: no overlapping files'); sys.exit()
u = sum(1 for k in common if len(hyp[k]) < len(ref[k]))
e = sum(1 for k in common if len(hyp[k]) == len(ref[k]))
o = sum(1 for k in common if len(hyp[k]) > len(ref[k]))
diff = sum(len(hyp[k]) - len(ref[k]) for k in common) / len(common)
n = len(common)
print('  spk-count over %d files: under %.1f%%  exact %.1f%%  over %.1f%%  mean_diff %+.2f'
      % (n, 100*u/n, 100*e/n, 100*o/n, diff))
print('  (control pos_weight=1.0: under 38.0%%  mean_diff -0.40 | paper: 32.4%%  -0.30)')
PYEOF
    return 0
}

run_lane () {
    local lane="$1" gpu="$2" posw="$3"
    local out="${BASELINE_OUT}_posw${posw}"
    local P="${LOG_DIR}/lane${lane}"
    mkdir -p "$P"

    log "LANE $lane on gpu $gpu (pos_weight=$posw)"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        preflight "$out" "lane $lane" || { log "LANE $lane ABORTED"; return 1; }
    fi
    if [ "$SCORE_ONLY" != "1" ]; then
        train_arm "posw${posw}" "$gpu" "$posw" "$out" "$P/train.log"
    fi
    score_arm "posw${posw}" "$gpu" "$out" "$P/infer.log"
    log "LANE $lane COMPLETE"
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _cvd <<< "$CUDA_VISIBLE_DEVICES"
    [ -n "${_cvd[0]:-}" ] && LANE_A_GPU="${_cvd[0]}"
    LANE_B_GPU="${_cvd[1]:-${_cvd[0]}}"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> lane A gpu $LANE_A_GPU, lane B gpu $LANE_B_GPU"
    unset CUDA_VISIBLE_DEVICES
fi

if [ ! -f "$CFG" ]; then
    echo "ERROR: $CFG not found -- run from the repo root." >&2; exit 1
fi

log "pos_weight queue starting"
log "  lane A gpu=$LANE_A_GPU pos_weight=$POSW_A"
log "  lane B gpu=$LANE_B_GPU pos_weight=$POSW_B"
log "  control (do not touch): $BASELINE_OUT"
log "  adapt init:             $ADAPT_INIT"
log "  patience=$PATIENCE max_epochs=$MAX_EPOCHS score_only=$SCORE_ONLY"
rm -f "$STOP_FILE"

pids=()
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "A" ]; then
    run_lane A "$LANE_A_GPU" "$POSW_A" & pids+=($!)
fi
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "B" ]; then
    run_lane B "$LANE_B_GPU" "$POSW_B" & pids+=($!)
fi
for pid in "${pids[@]}"; do wait "$pid"; done

log "pos_weight queue finished."
log "Compare against the control at MATCHED EPOCHS, not against its final"
log "number -- the control ran to 750. Its per-epoch checkpoints are local,"
log "so any epoch can be scored for a fair comparison point."
log "Pack results with: MODE=avg ./scripts/pack_results_from_server.sh"
