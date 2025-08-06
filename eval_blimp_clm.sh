#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

accelerate launch lm_eval --model hf-auto \
    --model_args pretrained=$MODEL_PATH,backend=causal,trust_remote_code=True,tokenizer=gpt2 \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json