#!/bin/sh

python src/merge_pseudo_and_back_translation.py \
    --pseudo-labels-path data/pseudo_labels/pseudo_labels_xl.csv \
    --back-translation-path data/generated_translation_like_english_10k_fold0/generated_translation_like_english_back_translated_xl.csv \
    --output-path data/pseudo_synthetic_merged/train_xl.csv \
    --pseudo-source-dataset pseudo_labels/pseudo_labels_xl \
    --synthetic-source-dataset generated_translation_like_english_10k_fold0/generated_translation_like_english_back_translated_xl
python src/train.py \
    --model outputs/finetune/xl-evacun-ft/best_model \
    --num-train-epochs 2 \
    --train_path data/pseudo_synthetic_merged/train_xl.csv \
    --train-tokenize-max-length 1024 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --full-fit \
    --run-name xl-pseudo-bt-pretrain \
    --apply_cut_if_too_long \
    --gradient_checkpointing \
    --report-to wandb
