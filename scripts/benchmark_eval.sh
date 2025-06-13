#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1

# Path to the models YAML file
MODELS_YAML="config/market_models.yaml"
DATASET_PATH="HuggingFaceH4/MATH-500"
OUTPUT_DIR="./results/math-500/reasoning"
PASS_AT_K=6

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
    
    python run_reasoner.py \
    --model_name "$model_name" \
    --nick_name "$nick_name" \
    --tokenizer_name "$model_name" \
    --dataset_name_or_path $DATASET_PATH \
    --split_name "test" \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_tokens 16384 \
    --temperature 0.6 \
    --top_p 1.0 \
    --top_k -1 \
    --pass_at_k $PASS_AT_K

done 