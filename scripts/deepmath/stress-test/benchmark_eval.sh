#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=2,3

# Hyperparameters
DATASET_NAME="deepmath"
PASS_AT_K=1
# Paths
MODELS_YAML="config/deepmath_models.yaml"
DATASET_PATH="./data/$DATASET_NAME"
OUTPUT_DIR="./results/$DATASET_NAME/benchmark"

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
    for enable_thinking in True False; do
        echo "Running model: $nick_name (model_name: $model_name) with enable_thinking=$enable_thinking"

        python benchmark_eval.py \
        --model_name "$model_name" \
        --nick_name "$nick_name" \
        --tokenizer_name "$model_name" \
        --dataset_name_or_path $DATASET_PATH \
        --split_name "test" \
        --output_dir $OUTPUT_DIR \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.75 \
        --dtype bfloat16 \
        --max_tokens 16384 \
        --temperature 0.6 \
        --top_p 1.0 \
        --top_k -1 \
        --pass_at_k $PASS_AT_K
    done
done 


python benchmark_eval.py \
--model_name aochongoliverli/Qwen2.5-1.5B-DeepMath-level1-4-40k-all_rollouts-sft-stage0-step-1890 \
--nick_name Qwen2.5-1.5B-DeepMath-level1-4-sft-stage0-step1890 \
--tokenizer_name aochongoliverli/Qwen2.5-1.5B-DeepMath-level1-4-40k-all_rollouts-sft-stage0-step-1890 \
--dataset_name_or_path ./data/deepmath \
--split_name test \
--output_dir ./results/deepmath/benchmark \
--tensor_parallel_size 2 \
--gpu_memory_utilization 0.75 \
--dtype bfloat16 \
--max_tokens 16384 \
--temperature 0.6 \
--top_p 1.0 \
--top_k -1 \
--pass_at_k 1 \
--max_num_batched_tokens 32768