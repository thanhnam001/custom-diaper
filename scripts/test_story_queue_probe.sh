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

# A python that does nothing and exits 0: no report line, no VRAM, so every
# downstream parse comes back empty. That is the worst case for the probe.
cat > "$T/bin/python" <<'STUB'
#!/bin/bash
exit 0
STUB
chmod +x "$T/bin/python"
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

PROBE_SECONDS=3
pass=0 fail=0

echo "TEST: probe with a no-op python reports cleanly and returns non-zero"
out="$(probe_stage probe_demo 0 "$T/p.yaml" 2>&1)"; rc=$?
echo "$out"
if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "NO REPORT LINE"; then
    echo "  ok   degraded gracefully (rc=$rc)"; pass=$((pass+1))
else
    echo "  FAIL expected a NO REPORT LINE message and rc!=0 (got rc=$rc)"
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
