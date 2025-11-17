#!/bin/bash
set -ex

# -------- static bits you rarely touch --------
DATASET_NAME="allcode"
SAMPLE_K=4
DATASET_PATH="./data/${DATASET_NAME}"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

# Define models as array of "model_name,nick_name" pairs
MODELS_NICK=(
    # "Qwen/Qwen3-235B-A22B-Thinking-2507,Qwen3-235B-A22B-2507"
    "deepseek-ai/DeepSeek-R1-0528,DeepSeek-R1-0528"
    # "Qwen/Qwen3-235B-A22B,Qwen3-235B-A22B"
    # "Qwen/QwQ-32B,QwQ-32B-teacher"
    # "Qwen/Qwen3-32B,Qwen3-32B-teacher"
)

# Loop through each model
for model_info in "${MODELS_NICK[@]}"; do
    IFS=, read -r model_name nick_name <<< "$model_info"
    echo "Running model: $nick_name (model_name: $model_name)"

    python benchmark_eval_api.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --dtype bfloat16 \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --sample_k $SAMPLE_K \
    --client_name 'deepinfra'
done 