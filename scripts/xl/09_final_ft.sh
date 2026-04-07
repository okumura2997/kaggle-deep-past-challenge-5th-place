#!/bin/sh

for fold in 0 1 2 3 ; do
    RUN_NAME="xl-final-ft/fold${fold}"
    python src/train_with_context.py \
            --model outputs/finetune/xl-pseudo-bt-pretrain/best_model \
            --train-processed-path data/extract_unified/all_pairs_final.csv \
            --train-tokenize-max-length 2048 \
            --per-device-train-batch-size 8 \
            --gradient-accumulation-steps 2 \
            --per-device-eval-batch-size 32 \
            --warmup-ratio 0.05 \
            --learning-rate 1e-4 \
            --max-steps 2200 \
            --eval-steps 200 \
            --eval-strategy steps \
            --sentence-concat-prob 0 \
            --n-folds 4 \
            --gradient-checkpointing \
            --fold "${fold}" \
            --run-name "${RUN_NAME}" \
            --report-to wandb
done
