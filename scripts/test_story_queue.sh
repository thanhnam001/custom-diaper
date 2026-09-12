#!/bin/bash
# Offline tests for scripts/run_4gpu_story_queue.sh.
#
#   ./scripts/test_story_queue.sh
#
# Runs on any machine -- no GPU, no server data, no real training. It sources
# the queue's helper functions and drives train_stage with a stub `python`
# that only records the arguments it was handed.
#
# WHAT IT ACTUALLY CHECKS, and why each one matters:
#
#   1. a fresh stage warm-starts    -- does NOT pass --init-model-path '',
#                                      so average_checkpoints() runs and the
#                                      arm actually inherits its adapt weights
#   2. a resumable stage does NOT   -- THE init_model_path TRAP. train.py runs
#      warm-start                      `if init_model_path != '': average_
#                                      checkpoints(...)` unconditionally,
#                                      BEFORE checking output_path for a
#                                      resumable checkpoint. A resume that
#                                      leaves init_model_path set will crash
#                                      on startup if the adapt dir is gone,
#                                      even though the resume never needed it.
#   3. missing init weights refuse  -- training from random init silently
#                                      answers a different question
#   4. a completed stage is skipped -- resumability across driver restarts
#   5. STOP is honoured             -- interruptibility
#
# Exits non-zero on the first failure.

set -u
cd "$(dirname "$0")/.." || exit 1

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
export LOG_DIR="$T/logs"
export USE_CONDA_RUN=0
mkdir -p "$LOG_DIR"

# Stub python: record argv, succeed. Must be found before the real python.
mkdir -p "$T/bin"
# Each arg is recorded as [arg] so that an EMPTY argument is still visible --
# `echo "$@"` collapses `--init-model-path ''` into a trailing space and the
# trap test below cannot see it.
cat > "$T/bin/python" <<'STUB'
#!/bin/bash
for a in "$@"; do printf '[%s]' "$a" >> "$CALLS"; done
echo >> "$CALLS"
exit 0
STUB
chmod +x "$T/bin/python"
export PATH="$T/bin:$PATH"
export CALLS="$T/calls.txt"
: > "$CALLS"

# shellcheck source=/dev/null
source scripts/run_4gpu_story_queue.sh || { echo "FAIL: could not source"; exit 1; }

pass=0 fail=0
ok   () { echo "  ok   $1"; pass=$((pass+1)); }
bad  () { echo "  FAIL $1"; fail=$((fail+1)); }

mkcfg () {  # mkcfg <file> <output_path> <init_model_path>
    cat > "$1" <<EOF
output_path: $2
init_model_path: $3
train_batchsize: 64
max_epochs: 500
subsampling: 10
num_frames: 600
EOF
}

echo "TEST 1+3: fresh stage with init weights present warm-starts"
mkcfg "$T/a.yaml" "$T/out_a" "$T/init_a"
mkdir -p "$T/init_a"; touch "$T/init_a/checkpoint_100.tar"
: > "$CALLS"
train_stage t1 0 "$T/a.yaml" >/dev/null 2>&1
if grep -qF -- "[--init-model-path][]" "$CALLS" 2>/dev/null; then
    bad "fresh stage must NOT blank init-model-path (it needs the warm start)"
elif grep -q "train.py" "$CALLS"; then
    ok "fresh stage ran without blanking init-model-path"
else
    bad "fresh stage did not invoke train.py at all"
fi

echo "TEST 2: resumable stage skips the warm start (the init_model_path trap)"
mkcfg "$T/b.yaml" "$T/out_b" "$T/init_gone"     # init dir deliberately absent
mkdir -p "$T/out_b/models"; touch "$T/out_b/models/checkpoint_207.tar"
: > "$CALLS"
train_stage t2 0 "$T/b.yaml" >/dev/null 2>&1
if grep -qF -- "[--init-model-path][]" "$CALLS" 2>/dev/null; then
    ok "resume passed --init-model-path '' despite the adapt dir being gone"
else
    bad "resume did NOT blank init-model-path -- would crash in average_checkpoints"
fi

echo "TEST 3b: fresh stage with MISSING init weights refuses to run"
mkcfg "$T/c.yaml" "$T/out_c" "$T/init_missing"
: > "$CALLS"
train_stage t3 0 "$T/c.yaml" >/dev/null 2>&1
rc=$?
if [ $rc -ne 0 ] && ! grep -q "train.py" "$CALLS"; then
    ok "refused to train from random init (rc=$rc, train.py never invoked)"
else
    bad "ran anyway with no init weights (rc=$rc) -- would answer a different question"
fi

echo "TEST 4: a completed stage is skipped"
mark_stage t4
: > "$CALLS"
train_stage t4 0 "$T/a.yaml" >/dev/null 2>&1
if [ -s "$CALLS" ]; then
    bad "re-ran a stage already marked done"
else
    ok "skipped the stage marked done"
fi

echo "TEST 5: STOP file is honoured"
: > "$STOP_FILE"
: > "$CALLS"
train_stage t5 0 "$T/a.yaml" >/dev/null 2>&1
if [ -s "$CALLS" ]; then
    bad "started a stage while STOP was present"
else
    ok "honoured STOP"
fi
rm -f "$STOP_FILE"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
