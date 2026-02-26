#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASET_NAME="allmath"
INPUT_DIR="./results/${DATASET_NAME}/teacher_guide"
RESULTS_DIR="./results/${DATASET_NAME}/teacher_guide/ppl"

python analysis/compute_perplexity.py \
    --model_name "${MODEL}" \
    --nickname "${NICK}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    --max_batch_tokens 32768