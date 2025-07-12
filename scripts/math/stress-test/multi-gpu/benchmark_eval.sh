#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

# -------- static bits you rarely touch --------
DATASET_NAME="allmath"
SAMPLE_K=8
DATASET_PATH="./data/${DATASET_NAME}"
OUTPUT_DIR="./results/${DATASET_NAME}/benchmark"
# ----------------------------------------------

export CUDA_VISIBLE_DEVICES=$GPU_ID

python benchmark_eval.py \
  --model_name            "$MODEL" \
  --nick_name             "$NICK" \
  --tokenizer_name        "$MODEL" \
  --dataset_name_or_path  "$DATASET_PATH" \
  --split_name            test \
  --output_dir            "$OUTPUT_DIR" \
  --tensor_parallel_size  $(echo "$GPU_ID" | awk -F',' '{print NF}') \
  --gpu_memory_utilization 0.9 \
  --dtype                 bfloat16 \
  --max_tokens            16384 \
  --temperature           0.7 \
  --top_p                 1.0 \
  --top_k                -1 \
  --sample_k             "$SAMPLE_K" \
  --max_num_batched_tokens 8192 \
  --overwrite             True \