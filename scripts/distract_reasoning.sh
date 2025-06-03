set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/hendrycks_math/sample200"
OUTPUT_DIR="$EVAL_DIR/distraction"

MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    nick_name_lower="${nick_name,,}"
    
    # Handle base models by removing "-Base" suffix for dataset path
    if [[ "$nick_name_lower" == *"-base" ]]; then
        instruct_name="${nick_name%-Base}"
        DATASET_PATH="$EVAL_DIR/reasoning/${instruct_name}_correct.pickle"
    else
        DATASET_PATH="$EVAL_DIR/reasoning/${nick_name}_correct.pickle"
    fi

    echo "Running Distraction Experiment:"
    echo "  Model: ${model_name}, ${nick_name}"
    echo "  Dataset: ${DATASET_PATH}"
    echo "  Output Directory: ${OUTPUT_DIR}"
    echo "-------------------------------------"

    # Check if this is a DeepMath model and adjust max_tokens
    if [[ "$nick_name_lower" == *"deepmath"* ]]; then
        MAX_TOKENS=4096
        echo "  DeepMath model detected: Setting max_tokens to ${MAX_TOKENS}"
    else
        MAX_TOKENS=16384
    fi
    
    python distract_thinking.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 0.9 \
        --top_k 32 \
        --num_distract_candidates 10 \
        --granularity 20

done 