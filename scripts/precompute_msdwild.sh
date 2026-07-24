#!/bin/bash
set -e

# MSDWild layout: all wavs in one folder, one combined *.rttm file per
# split (paper's few.train -> train, many.val -> dev, few.val -> test here;
# "test" must be few.val since infer_16k_10attractors.yaml reads kaldi/test
# and the README's reported MSDWild numbers are for the few.val split).

WAV_DIR=/data/ocr/namvt17/dataset/diarization/msdwild_wavs
RTTM_TRAIN=/data/ocr/namvt17/dataset/diarization/msdwild/rttms/few.train.rttm   # e.g. few.train.rttm
RTTM_DEV=/data/ocr/namvt17/dataset/diarization/msdwild/rttms/many.val.rttm     # e.g. many.val.rttm
RTTM_TEST=/data/ocr/namvt17/dataset/diarization/msdwild/rttms/few.val.rttm      # e.g. few.val.rttm
KALDI_DIR=/data/ocr/namvt17/dataset/diarization/msdwild/kaldi
PRECOMPUTE_DIR=/data/ocr/namvt17/dataset/diarization/msdwild_precompute_6000frames

declare -A RTTMS=( [train]=$RTTM_TRAIN [dev]=$RTTM_DEV [test]=$RTTM_TEST )
for split in train dev test; do
    python diaper/common_utils/prepare_kaldi_data_dir.py \
        --wav-dir $WAV_DIR \
        --rttm-file ${RTTMS[$split]} \
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
