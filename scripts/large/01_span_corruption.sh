#!/bin/sh

python src/pretrain.py \
    --model google/byt5-large \
    --run_name large-span-corruption \
    --gradient-checkpointing 