#!/bin/sh

python src/train.py \
    --train_path data/evacun/train_processed.csv \
    --model outputs/pretrain/large-span-corruption/best_model \
    --num-train-epochs 15 \
    --train-tokenize-max-length 512 \
    --per-device-train-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --fold 0 \
    --run-name large-evacun-ft \
    --gradient-checkpointing \
    --report-to wandb