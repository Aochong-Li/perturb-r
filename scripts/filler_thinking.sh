#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1

# Path to the models YAML file
MODELS_YAML="config/market_models.yaml"
DATASET_PATH="./data/deepmath_7to9"
EVAL_DIR="./results/deepmath_7to9"
PASS_AT_K=3
FILLER_WORD="wait"
NUM_FILLER_TOKENS=(5000 10000 15000)
OVERWRITE=false

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
    echo "Filler Thinking Stress Test: $nick_name (model_name: $model_name)"

    python stress-test/fillter_thinking.py \
        --model_name "$model_name" \
        --nick_name "$nick_name" \
        --tokenizer_name "$model_name" \
        --results_dir $EVAL_DIR \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --dtype bfloat16 \
        --max_tokens 4096 \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --pass_at_k $PASS_AT_K \
        --filler_word "$FILLER_WORD" \
        --num_filler_tokens ${NUM_FILLER_TOKENS[@]} \
        --overwrite $OVERWRITE
done 