set -ex

INPUT_DIR="./results/allcode/teacher_guide"
OUTPUT_DIR="./results/allcode/teacher_guide/teachability_contains_answer"

python stress-test/ablation/teachability_if_answer_cruxeval.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --stage all \
    --temperature 0.7 \
    --max_tokens 512
