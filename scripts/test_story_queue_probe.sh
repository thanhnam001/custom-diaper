#!/bin/bash
# Offline test for the DRY_RUN probe in scripts/run_4gpu_story_queue.sh.
#
#   ./scripts/test_story_queue_probe.sh
#
# The probe has the most moving parts of anything in the queue (a background
# nvidia-smi sampler, a `timeout`-bounded train.py, an awk timestamper and two
# nested python calls), and it runs on a machine that may have neither a GPU
# nor the server data. This checks it DEGRADES GRACEFULLY instead of crashing
# or hanging under `set -u`: a stage that produces no training output must be
# reported as such and return non-zero, not take the driver down.

set -u
cd "$(dirname "$0")/.." || exit 1

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
export LOG_DIR="$T/logs"
export USE_CONDA_RUN=0
mkdir -p "$LOG_DIR/dryrun" "$T/bin"

# The stub python must fake ONLY the train.py invocation. probe_stage also
# calls python for real work -- `python - <<script` to derive sec/step and
# `python -c` for the hour/ramp arithmetic -- so the stub delegates those to
# the real interpreter. Without this the stub answers the sec/step call with
# its own fake training output, which looks like a hang.
REAL_PY="$(command -v python || command -v python3)"
export REAL_PY
mk_stub () {  # mk_stub <<'BODY' ... BODY   -- body runs for train.py calls
    {
        echo '#!/bin/bash'
        echo 'case "${1:-}" in'
        echo '  -|-c) exec "$REAL_PY" "$@" ;;'
        echo 'esac'
        cat
    } > "$T/bin/python"
    chmod +x "$T/bin/python"
}

# A python that does nothing and exits 0: no report line, so every
# downstream parse comes back empty. That is the worst case for the probe.
mk_stub <<'STUB'
exit 0
STUB
export PATH="$T/bin:$PATH"

# shellcheck source=/dev/null
source scripts/run_4gpu_story_queue.sh || { echo "FAIL: could not source"; exit 1; }

cat > "$T/p.yaml" <<'YML'
output_path: /tmp/story_probe_nope
init_model_path:
train_batchsize: 64
max_epochs: 500
noam_warmup_steps: 50000
subsampling: 10
num_frames: 600
YML

PROBE_STEPS=4
PROBE_TIMEOUT=6
pass=0 fail=0

echo "TEST: probe with a no-op python reports cleanly and returns non-zero"
out="$(probe_stage probe_demo 0 "$T/p.yaml" 2>&1)"; rc=$?
echo "$out"
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "FAILED:"; then
    echo "  ok   degraded gracefully (rc=$rc)"; pass=$((pass+1))
else
    echo "  FAIL expected a FAILED: message and rc!=0 (got rc=$rc)"
    fail=$((fail+1))
fi

echo "TEST: an OOM in the log is named explicitly"
mk_stub <<'STUB'
echo "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
exit 1
STUB
out="$(probe_stage probe_oom 0 "$T/p.yaml" 2>&1)"; rc=$?
echo "$out"
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "OUT OF MEMORY"; then
    echo "  ok   OOM identified rather than reported as a generic failure"
    pass=$((pass+1))
else
    echo "  FAIL expected OUT OF MEMORY in the line (got rc=$rc)"; fail=$((fail+1))
fi

echo "TEST: probe counts real steps, stops at PROBE_STEPS, derives the numbers"
# Emits train.py's actual report format, one 'step' every 0.5 s, and would
# run far past PROBE_STEPS if the probe did not stop it.
mk_stub <<'STUB'
for i in $(seq 1 400); do
    echo "[epoch 1] batch $i/1172 train: loss=0.5 DER=20.00%"
    sleep 0.5
done
STUB
out="$(PROBE_STEPS=6 PROBE_TIMEOUT=120 probe_stage probe_ok 0 "$T/p.yaml" 2>&1)"
rc=$?
echo "$out"
ok_line=1
printf '%s' "$out" | grep -q "steps/ep=1172" || { echo "  (no steps/ep=1172)"; ok_line=0; }
printf '%s' "$out" | grep -q "chunks=75008"  || { echo "  (no chunks=75008)"; ok_line=0; }
printf '%s' "$out" | grep -qE "sec/step=0\.[0-9]" || { echo "  (no plausible sec/step)"; ok_line=0; }
# Must stop AT the limit, not tens of steps past it: the stub emits a step
# every 0.2 s, so a loop that only looks every 2 s would overshoot badly.
stopped_at="$(printf '%s' "$out" | grep -oE '\[[0-9]+ steps\]' | grep -oE '[0-9]+')"
if [ -n "$stopped_at" ] && [ "$stopped_at" -ge 6 ] && [ "$stopped_at" -le 9 ]; then
    :
else
    echo "  (stopped at '${stopped_at:-?}' steps, wanted 6-9)"; ok_line=0
fi
printf '%s' "$out" | grep -qE "startup=[0-9]+s" || { echo "  (no startup figure)"; ok_line=0; }
if [ $rc -eq 0 ] && [ "$ok_line" -eq 1 ]; then
    echo "  ok   stopped on steps and reported steps/ep, chunks and sec/step"
    pass=$((pass+1))
else
    echo "  FAIL step-bounded probe did not report as expected (rc=$rc)"
    fail=$((fail+1))
fi

echo "TEST: a batch override is passed to train.py and labels its own row"
# Echo the batch train.py was actually handed, so the override is verified at
# the argv level rather than just in the printed row.
mk_stub <<'STUB'
b="?"
while [ $# -gt 0 ]; do
    if [ "$1" = "--train-batchsize" ]; then b="$2"; fi
    shift
done
for i in $(seq 1 40); do
    echo "[epoch 1] batch $i/1172 train: got_batchsize=$b"
    sleep 0.2
done
STUB
out="$(PROBE_STEPS=4 PROBE_TIMEOUT=60 probe_stage probe_sw 0 "$T/p.yaml" 48 2>&1)"
rc=$?
echo "$out"
argv_batch="$(grep -o 'got_batchsize=[0-9]*' "$LOG_DIR/dryrun_probe_sw_b48.log" \
              2>/dev/null | head -1)"
if [ $rc -eq 0 ] \
   && printf '%s' "$out" | grep -q "probe_sw_b48" \
   && printf '%s' "$out" | grep -q "batch=48" \
   && [ "$argv_batch" = "got_batchsize=48" ]; then
    echo "  ok   override reached train.py's argv and keyed its own log/row"
    pass=$((pass+1))
else
    echo "  FAIL override not applied (rc=$rc, argv saw '${argv_batch:-nothing}')"
    fail=$((fail+1))
fi

echo "TEST: probe with a missing config reports cleanly and returns non-zero"
out="$(probe_stage probe_missing 0 "$T/absent.yaml" 2>&1)"; rc=$?
echo "$out"
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "CONFIG MISSING"; then
    echo "  ok   reported the missing config (rc=$rc)"; pass=$((pass+1))
else
    echo "  FAIL expected CONFIG MISSING and rc!=0 (got rc=$rc)"; fail=$((fail+1))
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
