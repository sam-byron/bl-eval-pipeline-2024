#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

# accelerate launch lm_eval --model hf \
#     --model_args pretrained=$MODEL_PATH,backend=causal,trust_remote_code=True \
#     --tasks blimp_filtered,blimp_supplement \
#     --device cuda:0 \
#     --batch_size 1 \
#     --log_samples \
#     --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

accelerate launch lm_eval --model hf-mlm \
--model_args pretrained=$MODEL_PATH,backend="mlm",trust_remote_code=True \
--tasks blimp_filtered,blimp_supplement \
--batch_size 1 \
--log_samples \
--output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.
