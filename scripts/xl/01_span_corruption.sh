#!/bin/sh

python src/pretrain.py \
    --model google/byt5-xl \
    --run_name xl-span-corruption \
    --gradient-checkpointing
