#!/bin/sh

python src/generate_few_shot_pool.py 
python src/generate_translation_like_english.py \
    --few-shot-paths data/few_shot_pool.csv \
    --few-shot-exclude-folds 0 \
    --num-generations 10000 \
    --output-path data/generated_translation_like_english_10k_fold0/generated_translation_like_english.csv

