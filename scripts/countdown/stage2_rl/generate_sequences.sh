#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Hyperparameters
DATASET_NAME="countdown_train_stage2_level5_35K"
PASS_AT_K=1
# Paths
MODELS_YAML="config/market_models.yaml"
DATASET_PATH="./data/$DATASET_NAME"
OUTPUT_DIR="./results/$DATASET_NAME/stage2_rl_gen_seq"

MODEL_NAME="aochongoliverli/Qwen2.5-3B-sft-distill-countdown-level3-4-150"
NICK_NAME="Qwen2.5-3B-countdown-level4-5-stage1_rl"

python benchmark_eval.py \
--model_name $MODEL_NAME \
--nick_name $NICK_NAME \
--tokenizer_name $MODEL_NAME \
--dataset_name_or_path $DATASET_PATH \
--split_name "test" \
--output_dir $OUTPUT_DIR \
--tensor_parallel_size 8 \
--gpu_memory_utilization 0.85 \
--dtype bfloat16 \
--max_tokens 8192 \
--temperature 0.6 \
--top_p 1.0 \
--top_k -1 \
--pass_at_k $PASS_AT_K \
--max_num_batched_tokens 32768 \
--overwrite True \
--enable_thinking True