#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID
DATASET_NAME="allmath"
RESULTS_DIR="./results/${DATASET_NAME}/"


python stress-test/ablation/inject_distractor_multiple_model.py \
    --model_name "${MODEL}" \
    --nick_name "${NICK}" \
    --tokenizer_name "${MODEL}" \
    --results_dir "${RESULTS_DIR}" \
    --num_distractor_models 5 \
    --tensor_parallel_size $(echo "$GPU_ID" | awk -F',' '{print NF}') \
    --gpu_memory_utilization 0.9 \
    --dtype bfloat16 \
    --max_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --max_num_batched_tokens 32768