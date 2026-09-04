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

Two GPUs — split by lineage, not by dataset (MSDWild is 490 files per
lineage against RAMC's 43, so a dataset split is badly lopsided):

```bash
CUDA_VISIBLE_DEVICES=0 ONLY='2500h_baseline|paperlr|paperlr_ebf|fixednoam_cnf' \
    ./scripts/run_infer_adapt_2500h.sh
CUDA_VISIBLE_DEVICES=1 ONLY='fixednoam_ebf|spkcounting|headoff|overlaploss3' \
    ./scripts/run_infer_adapt_2500h.sh
```

Resumable: `infer.py` skips any recording whose RTTM already exists, and
the script skips a run outright once its output holds one RTTM per
`wav.scp` line. Ctrl-C costs only the recording in flight. A DER summary
table prints at the end.

Other knobs: `DATASETS=ramc` (RAMC only — 43 files, quick smoke test),
`MAX_CHECKPOINTS_TO_AVERAGE`, `NUM_THREADS`, `CFG_ROOT`, and the env
paths `DIAPER_ENV` / `DSCORE_SRC` / `DSCORE_ENV`.

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
- Two real inference runs on a 2-recording MSDWild subset, covering both
  encoder types and both postprocessing paths. RTTMs landed exactly where
  the scoring glob looks for them.
- **The one thing that could have failed silently:**
  `average_checkpoints` loads with `strict=True`, so an adapt checkpoint
  carrying a counting head the infer config does not declare would hard-
  fail mid-queue. All 8 adapt checkpoints were compared key-for-key and
  shape-for-shape against their own finetuned checkpoints — identical
  across the board, so no `--allow-partial-warmstart` is needed.

## Caveats

- **Server only.** RAMC inference at `num_frames: -1` does not fit this
  machine's 6 GB GPU (0/43 files) — architectural, not a bug.
- The archive unpacks *next to*, not over, the finetune outputs already
  on the server: paths inside are relative to `experiments/10attractors`
  and every entry is `<experiment>/models/checkpoint_NN.tar`.
- Output goes to `<experiment>/{msdwild,ramc}_test_adapt_pred/`, a new
  directory per experiment, so nothing existing is touched.
