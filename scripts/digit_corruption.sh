set -ex

MODELS_YAML="config/market_models.yaml"
EVAL_DIR="./results/hendrycks_math/sample200/reasoning"
OUTPUT_DIR="$EVAL_DIR/digit_corruption"
MAX_TOKENS=16384
MODELS_INFO=$(python -c "
import yaml
with open('$MODELS_YAML', 'r') as f:
    data = yaml.safe_load(f)
for model in data['models']:
    print(f\"{model['model_name']},{model['nick_name']}\")
")

# Loop through each model
echo "$MODELS_INFO" | while IFS=, read -r model_name nick_name; do
    # Convert nick_name to lowercase for case-insensitive comparison
    nick_name_lower="${nick_name,,}"
    
    # Handle base models by removing "-Base" suffix for dataset path
    if [[ "$nick_name_lower" == *"-base" ]]; then
        instruct_name="${nick_name%-Base}"
        DATASET_PATH="$EVAL_DIR/reasoning/${instruct_name}_correct.pickle"
    else
        DATASET_PATH="$EVAL_DIR/reasoning/${nick_name}_correct.pickle"
    fi
    echo "Running Digit Corruption Experiment:"
    echo "  Model: ${model_name}, ${nick_name}"
    echo "  Dataset: ${DATASET_PATH}"
    echo "  Output Directory: ${OUTPUT_DIR}"
    echo "-------------------------------------"


    python digit_corruption.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 0.9 \
        --top_k 32 \
        --corrupt_type "answer_digit" \
        --continue_thinking False
    
    python digit_corruption.py \
        --model_name "${model_name}" \
        --nick_name "${nick_name}" \
        --tokenizer_name "${model_name}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --dtype bfloat16 \
        --max_tokens $MAX_TOKENS \
        --temperature 0.6 \
        --top_p 0.9 \
        --top_k 32 \
        --corrupt_type "midway" \
        --continue_thinking True
        
done