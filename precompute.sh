python diaper/common_utils/precompute_features.py \
    /mnt/d/Python/Master/repos/EEND_dataprep/v2/LibriSpeech/datasets/v1_10hours/--no-use-rirs--use-noises/train/data \
    ./test_10h \
    --chunk-size 6000 \
    --frame-size 400 \
    --frame-shift 160 \
    --sampling-rate 16000 \
    --feature-dim 40 \
    --input-transform logmel_meannorm \
    --n-speakers 10 \
    --use-last-samples \
    --num-workers 8 \
    --tail-subsampling 10