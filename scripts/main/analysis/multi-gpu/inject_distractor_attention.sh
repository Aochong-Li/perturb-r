#!/usr/bin/env bash
set -ex
# USAGE: ./eval_one.sh <gpu_id> <model_name> <nick_name>

GPU_ID=$1
MODEL=$2
NICK=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASET_NAME="allmath"
INPUT_DIR="./results/${DATASET_NAME}/inject_distractor"
RESULTS_DIR="./results/${DATASET_NAME}/inject_distractor/attention_scores"

python analysis/attention_scores.py \
    --model_name "${MODEL}" \
    --nickname "${NICK}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${RESULTS_DIR}"