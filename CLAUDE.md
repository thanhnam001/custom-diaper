# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DiaPer: a PyTorch implementation of end-to-end neural speaker diarization using
Perceiver-based attractors (https://arxiv.org/pdf/2312.04324.pdf). It's a research
codebase (originating from Hitachi's EEND / BUT's DiaPer), not a library — there is
no test suite, linter config, or package build. "Correctness" here means numerical/
shape correctness of tensors flowing through training and inference, verified by
actually running the scripts against data, not by unit tests.

## Environment

Conda env, Python 3.7-era pins (see README.md `## Getting started` for the exact
`pip`/`conda install` sequence — `torch==1.10.0+cu113`, `librosa`,
`safe_gpu`, `yamlargparse==1.31.1`, etc.). There is no `requirements.txt` or
`pyproject.toml`; the README install block is the source of truth for dependencies.
There is no `transformers` dependency: the Perceiver encoder this repo used to
pull it in for (via a fork, `pip install git+https://github.com/fnlandini/
transformers`) is vendored natively at `diaper/backend/perceiver.py`.

On this machine, the env already exists as a path-based conda env at
`../Master/repos/DiaPer/.diaper_env` (relative to this repo root, i.e.
`D:\Python\Master\repos\DiaPer\.diaper_env`) — use it for every `python
diaper/...` command in this repo rather than reinstalling. Activate with
`conda activate D:/Python/Master/repos/DiaPer/.diaper_env` (path-based envs
aren't in the named list `conda env list` shows first).

## Evaluation (scoring RTTMs)

DER is **not** computed by anything in this repo directly for final reporting —
`diaper/infer.py`/`infer_single_file.py` only produce RTTMs (see "Outputs"
below). Scoring those RTTMs against ground truth is done with
[dscore](https://github.com/nryant/dscore), checked out locally at
`../Master/repos/dscore` (relative to this repo root), using its own conda env
named `dscore`:

```bash
conda activate dscore
cd ../Master/repos/dscore
python score.py -r <gt_rttm...> -s <pred_rttm...>
# or, for many files: python score.py -R ref.scp -S sys.scp
```

See that repo's README.md for the full flag set (`-u`/UEM scoring regions,
`--collar`, `--ignore_overlaps`, `--n_digits`, `--table_format`, etc.);
`score.py`'s own docstring has more detail.

**Collar convention matters and differs by dataset** (confirmed against Table
II of the DiaPer paper, `DiaPer End-to-End Neural Diarization with
Perceiver-Based Attractors.pdf` at this repo's root — see "Background reading"
below):
- **MSDWiLD**: collar **0.25 s** (`--collar 0.25`).
- **RAMC**: collar **0 s** (dscore's default, no `--collar` flag needed).

The paper's inference-time post-processing is tied to that same collar choice
(Section IV.D) and matters for reproducing its numbers: when the forgiveness
collar is 0 s (RAMC), no median filter is applied and inference is run with
**5-frame subsampling** in the frame encoder instead of the usual 10, for
finer output resolution; when a collar is used (MSDWiLD, 0.25 s), a
**median filter with window 11** is applied over the speech activity output
instead. Match `infer.py`'s `--median-window-length`/subsampling flags to
whichever dataset you're scoring, not just the DER-scoring `--collar` flag.

## Commands

All commands are run from the repo root.

```bash
# Sanity check that the environment/setup works end-to-end (downloads nothing;
# uses examples/IS1009a.wav and a bundled pretrained model)
./run_example.sh

# Train (also used for adaptation/fine-tuning, just point --init-model-path at
# the model to adapt; see examples/finetune_adaptedmorespeakers.yaml)
python diaper/train.py -c examples/train_2speakers.yaml

# Inference over a whole Kaldi-style data dir
python diaper/infer.py -c examples/infer.yaml

# Inference on a single wav file (no Kaldi data dir needed)
python diaper/infer_single_file.py -c examples/infer_16k_10attractors.yaml \
    --wav-dir examples --wav-name IS1009a --models-path <dir> --rttms-dir <dir>

# Precompute mel-spectrogram features once, then train/eval repeatedly against
# the cache (see "Precomputed features" below)
python diaper/common_utils/precompute_features.py <kaldi_data_dir> <output_dir> \
    --chunk-size 6000 --frame-size 400 --frame-shift 160 --sampling-rate 16000 \
    --feature-dim 40 --input-transform logmel_meannorm --n-speakers 10 \
    --use-last-samples --num-workers 8

# Verify a precomputed cache reproduces KaldiDiarizationDataset bit-for-bit
# (see verify.sh for the exact flag translation between the two datasets)
python verify.py <kaldi_data_dir> <precomputed_dir> --chunk-size 600 \
    --frame-size 400 --frame-shift 160 --sampling-rate 16000 --feature-dim 40 \
    --input-transform logmel_meannorm --n-speakers 10 --subsampling 10 \
    --use-last-samples --num-items 700
```

There are no automated tests. `verify.py` is the closest thing to one: it
diffs `KaldiDiarizationDataset` items against `PrecomputedKaldiDiarizationDataset`
items and exits non-zero on mismatch — run it after touching feature/precompute
code. (It currently has a stray `breakpoint()` in its comparison loop —
remove that before running non-interactively.)

All `*.yaml` configs are `yamlargparse` configs consumed via `-c`; every CLI flag
in `parse_arguments()` in the corresponding script can also be overridden on the
command line after `-c config.yaml`. Paths inside example configs like
`<output directory>` / `<train Kaldi data directory>` are placeholders you must
fill in — they are not resolved automatically.

## Local data layout

On this machine, real-data corpora and their precomputed feature caches are
under `database/` (repo-local, gitignored):
- `database/msdwild/kaldi/{train,dev,test}` — Kaldi-style data dirs;
  `database/msdwild/rttms/{all,few.train,few.val,many.val}.rttm` — reference
  RTTMs (dscore `-r`/`-R` input). Collar for scoring: 0.25 s (see
  "Evaluation" above).
- `database/ramc/kaldi/{train,dev,test}` — Kaldi-style data dirs (plus the raw
  `DataPartition`/`KeywordList`/`MDT2021S003` corpus dirs RAMC ships with).
  Collar for scoring: 0 s.
- `database/msdwild_precompute_6000frames/{train,dev}` and
  `database/ramc_precomputed_6000frames/{train,dev}` — outputs of
  `precompute_features.py` (chunk-size 6000 raw frames), ready to point
  `--train-precomputed-dir`/`--valid-precomputed-dir` at directly.
- `database/` also has `LibriSpeech`, `callhome_*`, `voxconverse`,
  `simulated_rirs_16k`, `musan`-derived noise (`data/musan_bgnoise`), etc. for
  building simulated conversations (SC) or other corpora referenced in
  `models/*/*.yaml` configs.

Synthetic/simulated-conversation (SC) data — the large precomputed sets used
to pretrain/adapt models before real-data fine-tuning — lives outside the
repo at `E:/datasets`. Directory names there mirror the ones used on the
original training server (e.g. `v1_300hours_max10spks`,
`v1_500hours_fixed_2spks_precompute`,
`diaper_precompute_300h_maximum_10spks_24000frames`), but aren't guaranteed
to be a 1:1 match for every path referenced in old configs/comments (which
often point at server-only paths like `/data/ocr/namvt17/...` or
`/kaggle/input/...`). **If a config or comment references a synthetic dataset
that doesn't have an obvious match under `E:/datasets`, ask before assuming
which local directory (if any) corresponds to it** — don't guess a
substitute or fabricate a path.

## Background reading

`DiaPer End-to-End Neural Diarization with Perceiver-Based Attractors.pdf`
(repo root) is the original paper this codebase implements
(arXiv:2312.04324) — read it for the method rationale, training-recipe
details (SC generation, curriculum of 2-speaker → multi-speaker → in-domain
fine-tuning, LR/warmup choices in Table III), and the full evaluation-data
table (Table II — per-dataset collar, speaker-count range, hours) referenced
in "Evaluation" above.

The fresh, unmodified upstream repo this one was forked/customized from is
checked out locally at `../Master/repos/DiaPer` (relative to this repo root,
i.e. `D:\Python\Master\repos\DiaPer`) — use it as a reference to diff against
when investigating whether a discrepancy (metric calculation, loss, config
default, etc.) comes from an intentional local customization or is a
pre-existing part of the original code.

## Architecture

### Pipeline

`Kaldi-style data dir` (wav.scp/segments/utt2spk/spk2utt/rttm/reco2dur, see
`examples/prepare_data_dir.sh`) → **dataset** (STFT → log-mel → splice → subsample,
optionally speaker-labeled) → **model** (frame encoder → Perceiver latents →
attractors → per-frame activation logits) → **loss** (PIT-matched BCE + several
auxiliary losses) → checkpoints → **inference** (attractor existence threshold +
median filter → RTTM).

### Model (`diaper/backend/models.py`)

`AttractorPerceiver` is the only model type (`get_model()` asserts on this). Key
flow in `forward()`:
1. A frame encoder (either a plain `nn.Linear`, or, if
   `use_frame_selfattention`, a stack of `frame_encoder_layers` self-attention +
   FFN blocks) turns spliced acoustic frames into frame embeddings `e`.
2. `get_attractors()` cross-attends a fixed set of learned latent vectors
   (`latent_attractors`) against the frame embeddings via a `PerceiverBlock`
   (chained `PerceiverEncoder`s, natively implemented in
   `diaper/backend/perceiver.py` — see "Model" note below), producing
   per-Perceiver-block
   latents which are mapped to attractors via `latents2attractors`
   (`dummy` = latents *are* attractors, requires `n_latents == n_attractors`;
   `linear` or `weighted_average` project `n_latents → n_attractors`).
3. If `condition_frame_encoder` is set, attractor-vs-frame dot products
   ("activation logits") are fed back to condition the next frame-encoder layer
   — attractor estimation and frame encoding are interleaved, not sequential.
4. The model returns everything at *every* frame-encoder layer and every
   Perceiver block (the `per_frameenclayer_*` / `per_prcvblock_*` tensors,
   stacked on their last axis) so that `intermediate_loss_frameencoder` /
   `intermediate_loss_perceiver` can supervise intermediate layers, not just
   the final output.
5. `torch.nn.DataParallel` always wraps the model (see `get_model()`); code that
   needs the raw module (e.g. `speaker_layer`) goes through `model.module.*`.

Speaker ID is an optional auxiliary head (`VanillaSpeakerLayer` or
`ArcfaceSpeakerLayer`) trained jointly via `speakerid_loss`/`speakerid-loss-weight`.

### Losses (`diaper/backend/losses.py`)

`pit_loss_multispk` does permutation-invariant matching (Hungarian algorithm via
`scipy.optimize.linear_sum_assignment`) between predicted and reference speakers
per-sequence before computing BCE — this is why attractor order is meaningless
and everything downstream (speaker-ID loss, metrics) re-derives the permutation
rather than assuming attractor `i` == reference speaker `i`. `-1` in label
tensors always means "padding, exclude from loss/metrics", distinct from `0`
("silence"). `get_loss()` is the per-layer entry point called once per
frame-encoder layer / Perceiver block when intermediate supervision is on
(see `train.py::compute_loss_and_metrics`).

### Datasets (`diaper/common_utils/`)

- `kaldi_data.py` / `features.py`: low-level Kaldi data-dir loading and the
  STFT → log-mel → splice → subsample feature pipeline.
- `diarization_dataset.py::KaldiDiarizationDataset`: computes features on the
  fly per `__getitem__` (STFT'd fresh for each chunk — chunking boundaries are
  decided in raw-frame domain in `__init__` first, features computed after).
  `chunk_size` here is in the **subsampled** domain; raw span = `chunk_size *
  subsampling`.
- `common_utils/precompute_features.py`: offline precompute of the mel
  spectrogram (stops right after `features.transform()`, before
  splice/subsample/specaugment/top-N speaker selection, since those are cheap
  and hyperparameter-dependent — see the module docstring for the full
  rationale). **`chunk_size` here is already in the RAW frame domain**, unlike
  `KaldiDiarizationDataset` — to reproduce a specific `KaldiDiarizationDataset`
  config exactly, multiply by that config's `subsampling` yourself.
  This file has its **own copy** of `KaldiData`/`load_wav_scp`/`stft`/etc.
  duplicated from `kaldi_data.py`/`features.py` rather than importing them —
  be aware when fixing bugs in one that the other likely needs the same fix.
  `--storage-format` chooses the on-disk layout of the cache it writes:
  `perfile` (default) writes one `{i:08d}.pkl` per chunk — simple, fine on
  local/NVMe disk. `shard` instead packs `--chunks-per-shard` chunks (default
  1000) into each of a handful of `shard_NNNNN.bin` files plus a byte-offset
  index in `meta.pkl`. Use `shard` when `precomputed_dir` lives on a network/
  parallel filesystem (NFS, Lustre, etc., e.g. a RunAI pod with no local
  NVMe) — there, a large SC corpus chunked into hundreds of thousands of
  files means the DataLoader opens ~one file per chunk per epoch, which
  hits the metadata server rather than disk throughput and shows up as
  epoch times that vary wildly with other tenants' load, independent of GPU
  speed. `common_utils/pack_shards.py` converts an existing `perfile` cache
  to `shard` **without recomputing features** (byte-copies each already-
  serialized chunk into shards) and spot-verifies the result before you
  delete the original — use it rather than re-running precompute from
  scratch on an expensive existing cache.
- `precomputed_diarization_dataset.py::PrecomputedKaldiDiarizationDataset`:
  drop-in replacement for `KaldiDiarizationDataset` that reads the precomputed
  cache and applies splice/subsample/specaugment/top-N-speaker-selection at
  load time, so those stay free-to-change knobs without re-running precompute.
  Reads either `storage_format` transparently, auto-detected from
  `meta.pkl` (missing key = `perfile`, for caches written before this
  existed) — no train/infer-side flag needed, just point
  `--*-precomputed-dir` at whichever cache directory you want.
- `verify.py` (repo root) cross-checks the two datasets produce identical
  output for equivalent configs — read its module docstring before using it,
  the `chunk_size`/`subsampling` translation between the two datasets is easy
  to get backwards.

### Two coexisting import conventions (know this before editing)

The codebase is mid-migration between two different ways of running it, and
files currently mix both:

- **Script-relative style** (`train.py`, `infer.py`, `infer_single_file.py`,
  `process_data.py`, `backend/models.py`, `backend/updater.py`): bare imports
  like `from backend.losses import ...` / `from common_utils.diarization_dataset
  import ...`. These assume the script is invoked as `python diaper/train.py`
  (which puts `diaper/` itself, not the repo root, on `sys.path`).
- **Package-qualified style** (`common_utils/diarization_dataset.py`,
  `common_utils/features.py`, `common_utils/precomputed_diarization_dataset.py`,
  and root-level `verify.py`): imports like `import diaper.common_utils.features
  as features`, which require the repo root (containing `diaper/__init__.py`)
  to be on `sys.path` — i.e. running as `python -m diaper.x` or a script that
  lives at the repo root, not `python diaper/train.py`.

These two styles are **not simultaneously satisfiable** with a single
`sys.path` setup: `python diaper/train.py` (per the README/`train.sh`) resolves
`train.py`'s own bare imports fine, but transitively breaks the moment it hits
`diarization_dataset.py`'s `import diaper.common_utils.features`, because the
repo root isn't on `sys.path` in that invocation. When touching import lines,
check which convention the rest of the file already uses and match it; when
adding new entry-point scripts, prefer the package-qualified style (like
`verify.py`) and run them from the repo root, since that's the direction the
in-progress migration (`diaper/__init__.py` was just added) is heading.

### Config surface

Every script's `parse_arguments()` defines the full set of valid hyperparameters
for that script — there's no shared config schema, so `train.py`, `infer.py` and
`process_data.py` each redeclare overlapping-but-not-identical argument lists
(e.g. inference has `estimate_spk_qty`/`estimate_spk_qty_thr`/`threshold`/
`median_window_length` for post-processing that training doesn't need). When
adding a new hyperparameter that affects the model or feature pipeline, it
usually needs to be added in more than one of these `parse_arguments()`
functions to stay usable end-to-end.

### Outputs

- Training writes checkpoints to `<output_path>/models/checkpoint_<epoch>.tar`
  and TensorBoard logs to `<output_path>/tensorboard`; it auto-resumes from the
  latest checkpoint found there.
- Inference writes one RTTM per recording under a directory encoding the
  postprocessing settings used (`rttms_dir/epochs.../spk_qty.../median.../rttms/`).
- `models/` at the repo root holds pretrained checkpoints (10/20-attractor
  variants, with/without fine-tuning) referenced by the `examples/infer_*.yaml`
  configs and `run_example.sh`.
