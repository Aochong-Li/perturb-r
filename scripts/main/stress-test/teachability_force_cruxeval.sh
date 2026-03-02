#!/bin/bash
set -ex

# CruxEval leakage detection via forced generation
# Tests whether teacher steers for CruxEval already contain enough
# information to predict the output without additional reasoning.
#
# Output goes to teachability_force_cruxeval/ (separate from force_code/)

export CUDA_VISIBLE_DEVICES=0
TP=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

DATASET_NAME="allcode"
INPUT_DIR="./results/${DATASET_NAME}/teacher_guide"
OUTPUT_DIR="./results/${DATASET_NAME}/teacher_guide/teachability_force_cruxeval"

# Same 7 models that have existing coding teacher_guide results
MODELS_INFO="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,R1-Distill-Qwen-1.5B
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,R1-Distill-Qwen-7B
deepseek-ai/DeepSeek-R1-Distill-Llama-8B,R1-Distill-Llama-8B
Qwen/Qwen3-1.7B,Qwen3-1.7B
open-thoughts/OpenThinker3-1.5B,OpenThinker3-1.5B
agentica-org/DeepScaleR-1.5B-Preview,DeepScaleR-1.5B-Preview
zwhe99/DeepMath-1.5B,DeepMath-1.5B"

echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    echo "=== Force CruxEval: $nick_name (model: $model_name) ==="

    python stress-test/ablation/teachability_force_cruxeval.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size ${TP} \
        --gpu_memory_utilization 0.9 \
        --dtype bfloat16 \
        --max_tokens 512 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k -1 \
        --max_num_batched_tokens 32768 \
        --n_eval_workers 32
done
