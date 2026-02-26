# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES=0,1
GPUS=(0 1)
MAX_TOKENS=8192

MODEL_LIST=(
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base"
)

# -------- round-robin launch --------
next=0
for MODEL in "${MODEL_LIST[@]}";do
    GPU=${GPUS[$next]}
    echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
    OUTPUT_DIR="./results/${MODEL}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR $GPU &

    next=$(( (next + 1) % ${#GPUS[@]} ))

    # if every GPU already busy, wait for one to finish
    while (( $(jobs -pr | wc -l) >= ${#GPUS[@]} )); do
        sleep 1
    done
done

wait
echo "✔ all models evaluated."