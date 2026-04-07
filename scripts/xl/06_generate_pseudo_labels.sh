#!/bin/sh

set -eu

mkdir -p data/pseudo_labels
python src/generate_published_texts_without_train.py --overwrite-output
python src/infer.py \
    --model outputs/finetune/xl-tablet-ft/fold0/best_model \
    --input-path data/published_texts_without_train_or_extracted.csv \
    --output-path data/pseudo_labels/pseudo_labels_xl.csv \
    --output-id-column oare_id \
    --per-device-eval-batch-size 4 \
    --add-transliteration
