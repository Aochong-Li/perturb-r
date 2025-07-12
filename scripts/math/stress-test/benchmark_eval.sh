#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# -------- static bits you rarely touch --------
MODELS_YAML="config/market_models.yaml"
DATASET_NAME="allmath"
SAMPLE_K=8
DATASET_PATH="./data/${DATASET_NAME}"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

# Use Python to extract model information from YAML
MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    echo "Running model: $nick_name (model_name: $model_name)"

    python benchmark_eval.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --sample_k $SAMPLE_K \
    --max_num_batched_tokens 8192 \
    --overwrite True
done 