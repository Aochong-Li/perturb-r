#!/bin/bash

# Example script to run cross-model distractor injection ablation study
# This script demonstrates how to test a model with distractors from other models

MODEL_NAME="/path/to/your/model"  # Change this to your model path
NICK_NAME="Qwen3-1.7B"            # Change this to match your model's pickle filename
TOKENIZER_NAME="/path/to/tokenizer"  # Change this to your tokenizer path
RESULTS_DIR="/share/goyal/lio/reasoning/eval/"  # Directory with inject_distractor results

# GPU and model settings
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
DTYPE="bfloat16"
MAX_TOKENS=32768
TEMPERATURE=0.6

# Distractor settings
NUM_DISTRACTOR_MODELS=5  # Randomly select 5 other models as distractor sources

# Run the cross-model ablation
python stress-test/ablation/inject_distractor_cross_model.py \
    --model_name "${MODEL_NAME}" \
    --nick_name "${NICK_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --results_dir "${RESULTS_DIR}" \
    --num_distractor_models ${NUM_DISTRACTOR_MODELS} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --dtype "${DTYPE}" \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --overwrite

echo "Cross-model distractor injection completed!"
echo "Results saved to: ${RESULTS_DIR}/inject_distractor_cross_model/${NICK_NAME}.pickle"
