#!/bin/bash
set -e

# RAMC layout: all wavs in one folder, one *.rttm file per recording,
# with the per-recording rttm files split across train/dev/test folders.

WAV_DIR=/data/ocr/namvt17/dataset/diarization/ramc/MDT2021S003/WAV
RTTM_ROOT=/data/ocr/namvt17/dataset/diarization/ramc/MDT2021S003/RTTM
KALDI_DIR=/data/ocr/namvt17/dataset/diarization/ramc/kaldi
PRECOMPUTE_DIR=/data/ocr/namvt17/dataset/diarization/ramc_precomputed_6000frames

for split in train dev test; do
    python diaper/common_utils/prepare_kaldi_data_dir.py \
        --wav-dir $WAV_DIR \
        --rttm-dir $RTTM_ROOT/$split \
        --output-dir $KALDI_DIR/$split
done

# Only train/dev need a precomputed cache (consumed by train.py's
# --train-precomputed-dir/--valid-precomputed-dir). test stays a plain
# kaldi-style dir since infer.py reads wavs directly, no cache involved.
# chunk-size = num_frames * subsampling from the finetune train.yaml
# (600 * 10 = 6000), see common_utils/precompute_features.py docstring.
for split in train dev; do
    python diaper/common_utils/precompute_features.py \
        $KALDI_DIR/$split \
        $PRECOMPUTE_DIR/$split \
        --chunk-size 6000 \
        --frame-size 400 \
        --frame-shift 160 \
        --sampling-rate 16000 \
        --feature-dim 40 \
        --input-transform logmel_meannorm \
        --use-last-samples \
        --num-workers 8
done
