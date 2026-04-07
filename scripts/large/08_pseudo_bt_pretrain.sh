#!/bin/sh

python src/merge_pseudo_and_back_translation.py
python src/train.py \
    --model outputs/finetune/large-evacun-ft/best_model \
    --num-train-epochs 2 \
    --train_path data/pseudo_synthetic_merged/train_large.csv \
    --train-tokenize-max-length 1024 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --full-fit \
    --run-name large-pseudo-bt-pretrain \
    --apply_cut_if_too_long \
    --gradient_checkpointing \
    --report-to wandb