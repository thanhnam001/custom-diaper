# Adapt-stage inference: 2500h lineages, MSDWild + RAMC

Runbook for `scripts/run_infer_adapt_2500h.sh` and
`scripts/pack_adapt_weights_for_server.sh`.

## Why

Every 2500h lineage has a published *finetuned* DER and none has an
*adapt-stage* one — the point right after multi-speaker adaptation on
synthetic data, before any MSDWild or RAMC finetuning. That is the column
the finetune-stage analysis keeps asking for: the adapt-stage gap to the
paper is ~1.4 DER and the finetuned gap ~3.2, so the lineages diverge
during finetuning — but without a per-architecture adapt row you cannot
say which of them arrived at the finetune stage already behind.

8 lineages x 2 datasets = 16 runs, all averaging `epochs 90-100`.

## Run it

On this machine, pack the weights (already done once; re-run only if the
adapt checkpoints change):

```bash
EXP=/d/Python/custom-diaper/experiments/10attractors \
OUT=/d/Python/custom-diaper/diaper_adapt_weights.tar \
    ./scripts/pack_adapt_weights_for_server.sh
```

Produces `diaper_adapt_weights.tar` — 1.9 GB, 80 checkpoints (the last 10
of each of the 8 adapt directories). Upload it, then on the server, from
the repo root:

```bash
git pull
tar -xf diaper_adapt_weights.tar \
    -C /data/ocr/namvt17/custom-diaper/experiments/10attractors

DRY_RUN=1 ./scripts/run_infer_adapt_2500h.sh    # check the plan first
./scripts/run_infer_adapt_2500h.sh
```

### The two datasets run on different hardware

This is the thing to get right, and it is not symmetric:

| | MSDWild | RAMC |
|---|---|---|
| device | GPU, inline | **CPU only** (`--gpu 0`) |
| files | 490 | 43 |
| cost | fits a V100 fine | **~120–130 GB RAM**, ~50 min per run |

Whole-recording RAMC inference (31-min median recordings, `subsampling: 5`,
`num_frames: -1`, O(T²) whole-recording self-attention) **does not fit a
V100 — not one, not two.** 2×V100 is 64 GB of VRAM against a ~130 GB
requirement. It runs on the CPU path only.

Worse, two *concurrent* RAMC runs would ask for ~250 GB and take the box
down rather than just the job. So every RAMC run takes a cross-shell mkdir
lock and they queue behind each other automatically, even across separate
shells. Eight serialized is ~6–7 h.

So: two GPUs are worth it for the **MSDWild half only** — the RAMC half
uses no GPU and serializes itself regardless.

```bash
CUDA_VISIBLE_DEVICES=0 DATASETS=msdwild \
    ONLY='2500h_baseline|paperlr|paperlr_ebf|fixednoam_cnf' \
    ./scripts/run_infer_adapt_2500h.sh
CUDA_VISIBLE_DEVICES=1 DATASETS=msdwild \
    ONLY='fixednoam_ebf|spkcounting|headoff|overlaploss3' \
    ./scripts/run_infer_adapt_2500h.sh
```

Then the CPU half, once, in its own shell. It needs ~130 GB free *on top
of* whatever the GPU lanes are using, so overlap it with the above only if
the box has that to spare:

```bash
DATASETS=ramc ./scripts/run_infer_adapt_2500h.sh
```

Not used, on purpose: `--fallback-cpu-oom`. Its CPU retry has no memory
ceiling and can trigger an OS-level OOM that kills unrelated processes
rather than just itself. For a runaway guard set `RAMC_MAX_INPUT_FRAMES`
instead — it *skips* over-long recordings, which makes the DER a subset
number, and the script warns when that happens.

### Everything else

Resumable: `infer.py` skips any recording whose RTTM already exists, and
the script skips a run outright once its output holds one RTTM per
`wav.scp` line. Ctrl-C costs only the recording in flight (the lock is
released on exit, and a lock whose owner died is reclaimed). A DER summary
table prints at the end.

Other knobs: `MAX_CHECKPOINTS_TO_AVERAGE`, `NUM_THREADS` (GPU runs),
`RAMC_CPU_THREADS` (default 8), `RAMC_INFER_DEVICE`, `RAMC_LOCK_TIMEOUT`,
`LOG_DIR` (holds the lock), `CFG_ROOT`, and the env paths `DIAPER_ENV` /
`DSCORE_SRC` / `DSCORE_ENV`.

`infer.py`'s `--gpu` is a **device flag**, not a process count like
`train.py`'s: `>= 1` means CUDA, anything lower means CPU.

`ONLY` is an **anchored** extended regex over the labels, not a substring
— `ONLY=fixednoam` matches nothing, `ONLY='fixednoam.*'` matches both.
The eight labels:

    2500h_baseline   paperlr       paperlr_ebf   fixednoam_cnf
    fixednoam_ebf    spkcounting   headoff       overlaploss3

## What each run does

Each run reuses that lineage's own `infer_{msdwild,ramc}*.yaml`, so the
architecture flags, test data dir and per-dataset postprocessing stay
identical to the run that produced its finetuned number and the two stay
comparable. Only two things are overridden:

    --models-path   <experiment>/models              (adapt, not finetuned)
    --rttms-dir     <experiment>/<ds>_test_adapt_pred

The experiment directory is *derived* from the config's own `models_path`
(which points at `<experiment>/models_finetuneXXX/models`), so the script
hardcodes no experiment paths and follows the configs if the server root
moves.

| label | config dir | encoder |
|---|---|---|
| `2500h_baseline` | `SC_LibriSpeech_2spk_2500h` | self-attention |
| `paperlr` | `..._2500h_paperlr` | self-attention |
| `paperlr_ebf` | `..._2500h_paperlr_ebranchformer` | E-Branchformer |
| `fixednoam_cnf` | `..._2500h_fixednoam_conformer_k31` | conformer |
| `fixednoam_ebf` | `..._2500h_fixednoam_ebf` | E-Branchformer |
| `spkcounting` | `..._kernel31_mlp_fresh_2500h_spkcounting` | conformer |
| `headoff` | `..._spkcounting_headoff` | conformer |
| `overlaploss3` | `..._spkcounting_overlaploss3` | conformer |

Per-dataset protocol, taken from the configs, not overridden:

| | MSDWild | RAMC |
|---|---|---|
| recordings | 490 | 43 |
| median window | 11 | 1 (none) |
| subsampling | 10 | 5 |
| dscore collar | 0.25 s | 0 s |

## Finetune coverage audit

All 44 finetune runs were checked, comparing each run's latest checkpoint
window against the epoch ranges that actually have RTTMs on disk.

**Every 2500h finetune run is already inferred at its latest
checkpoints.** All five gaps are in the older 300h lineages:

| experiment | run | checkpoints reach | last inference |
|---|---|---|---|
| `adapted1-10_300h_maximum5spks` | MSDWILD | ep 447 | `epochs390-400` |
| `adapted1-10_300h_maximum5spks` | RAMC | ep 149 | `epochs130-140` |
| `adapted1-10_conformer_conv_kernel3_300h_max5spks` | MSDWILD | ep 431 | `epochs390-400` |
| `adapted1-10_conformer_conv_kernel3_300h_max5spks` | RAMC | ep 149 | `epochs130-140` |
| `linear_nodiversity_adapted1-10_300h_max10spks` | MSDWILD | **0 ckpts** | never ran |

The first four are the superseded `maximum5spks` lineage — they kept
training past their last scoring. The fifth is an empty directory. None
were re-run; they are outside the 2500h set.

One near-miss ruled out:
`paperlr_ebf/models_finetuneMSDWILD_subsampling5` looks stale until you
notice its `epochs740-750` output lives in `msdwild_test_full_pred/` and
`msdwild_test_infersubsampling10_pred/`, not `msdwild_test_pred/`.

## Verified before committing

- `bash -n` on both scripts.
- Full 16-run `DRY_RUN` against local copies of the experiment dirs: all
  resolve to `epochs 90-100`, correct adapt dirs, correct per-dataset
  postprocessing.
- The DER column index (`awk $4`) against a real dscore log.
- Three real inference runs on a 2-recording subset, covering both encoder
  types, both postprocessing paths, and the `--gpu 0` CPU path. RTTMs
  landed exactly where the scoring glob looks for them.
- The RAMC lock, functionally: a lock held by a **live** pid blocks the
  run and reports `FAIL (RAMC lock timeout)` instead of proceeding; a lock
  held by a **dead** pid is reclaimed; the lock is released after a failed
  inference; and the exit trap does not steal a lock owned by another
  shell.
- **The one thing that could have failed silently:**
  `average_checkpoints` loads with `strict=True`, so an adapt checkpoint
  carrying a counting head the infer config does not declare would hard-
  fail mid-queue. All 8 adapt checkpoints were compared key-for-key and
  shape-for-shape against their own finetuned checkpoints — identical
  across the board, so no `--allow-partial-warmstart` is needed.

## Caveats

- **Server only, and RAMC is CPU-only even there.** RAMC whole-recording
  inference at `subsampling: 5` needs ~120–130 GB RAM; it does not fit a
  V100, let alone this laptop's 6 GB GPU (0/43 files). Architectural
  (O(T²) whole-recording self-attention, no chunking, by paper design),
  not a bug. See the memory note `diaper_ramc_infer_hardware_limit`.
- The archive unpacks *next to*, not over, the finetune outputs already
  on the server: paths inside are relative to `experiments/10attractors`
  and every entry is `<experiment>/models/checkpoint_NN.tar`.
- Output goes to `<experiment>/{msdwild,ramc}_test_adapt_pred/`, a new
  directory per experiment, so nothing existing is touched.
