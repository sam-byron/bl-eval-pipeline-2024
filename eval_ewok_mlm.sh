#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

accelerate launch lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,backend="mlm" \
    --tasks ewok_filtered \
    --device cuda:0 \
    --batch_size 128 \
    --log_samples \
    --output_path results/ewok/${MODEL_BASENAME}/ewok_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.