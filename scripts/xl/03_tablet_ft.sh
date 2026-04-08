#!/bin/sh

python src/train.py \
    --model outputs/finetune/xl-evacun-ft/best_model \
    --num-train-epochs 5 \
    --add-extracted-data \
    --extracted-data-path data/extract_excavation_translation_pairs_from_locations/translations_by_record_merged_processed_en.csv \
    --train_path data/train_processed.csv \
    --train-tokenize-max-length 1024 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --fold 0 \
    --run-name xl-tablet-ft/fold0 \
    --apply_cut_if_too_long \
    --gradient_checkpointing \
    --report-to wandb
