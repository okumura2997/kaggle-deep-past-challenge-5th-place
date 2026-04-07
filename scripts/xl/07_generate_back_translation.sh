#!/bin/sh

python src/infer.py \
    --model outputs/finetune/back-translation-xl-tablet-ft/fold0/best_model \
    --input-path data/generated_translation_like_english_10k_fold0/generated_translation_like_english.csv \
    --output-path data/generated_translation_like_english_10k_fold0/generated_translation_like_english_back_translated_xl.csv \
    --translation-direction english_to_akkadian \
    --output-id-column synthetic_id \
    --per_device_eval_batch_size 4 \
    --generation-num-beams 4 \
    --no-generation-do-sample \
    --add-translation \
    --write_output_per_batch
