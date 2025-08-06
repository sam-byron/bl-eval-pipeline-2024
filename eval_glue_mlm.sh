#!/bin/bash

# This script runs the GLUE benchmark on a given Hugging Face model.

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model_checkpoint>"
    exit 1
fi

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)
OUTPUT_DIR="results/glue/${MODEL_BASENAME}"

mkdir -p $OUTPUT_DIR

# Note: GLUE tasks are classification/regression tasks.
# lm-eval adapts them for generative models.
# Adjust batch_size based on your GPU memory.
accelerate launch lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,backend=mlm \
    --tasks glue \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path ${OUTPUT_DIR}/glue_results.json