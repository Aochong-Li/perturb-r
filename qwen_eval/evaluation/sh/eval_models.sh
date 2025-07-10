# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES=0
MAX_TOKENS=16384

MODEL_LIST=(
    "aochongoliverli/Qwen2.5-Math-1.5B_drgrpo_parquet_rollout_8_max_length_3000_epoch_19"
)
for MODEL in "${MODEL_LIST[@]}";do
    echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
    OUTPUT_DIR="/share/goyal/lio/reasoning/eval/benchmarks/deepmath/${MODEL}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR
done