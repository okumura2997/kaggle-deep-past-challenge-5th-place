#!/bin/sh

python src/train.py \
    --train_path data/evacun/train_processed.csv \
    --model google/byt5-xl \
    --translation_direction english_to_akkadian \
    --num-train-epochs 10 \
    --train-tokenize-max-length 512 \
    --per-device-train-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --fold 0 \
    --run-name back-translation-xl-evacun-ft \
    --gradient-checkpointing \
    --report-to wandb
