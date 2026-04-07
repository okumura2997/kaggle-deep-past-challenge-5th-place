#!/bin/sh

python src/average_fold_best_models.py \
    --checkpoint-dirs \
        outputs/finetune/xl-final-ft/fold0/best_model \
        outputs/finetune/xl-final-ft/fold1/best_model \
        outputs/finetune/xl-final-ft/fold2/best_model \
        outputs/finetune/xl-final-ft/fold3/best_model \
    --output-dir outputs/finetune/xl-final-ft/averaged
