#!/usr/bin/env bash
set -euo pipefail

MODELS_YAML="./config/local_models.yaml"
GPUS=(0 1 2 3 4 5 6 7)   # edit if you want a subset

# -------- load (model,nick) pairs into an array --------
mapfile -t MODEL_LINES < <(
python - <<'PY' "$MODELS_YAML"
import sys, yaml
for m in yaml.safe_load(open(sys.argv[1]))['models']:
    print(m['model_name'], m['nick_name'])
PY
)

# -------- round-robin launch --------
next=0
for line in "${MODEL_LINES[@]}"; do
    read -r MODEL NICK <<<"$line"
    GPU=${GPUS[$next]}

    ./scripts/math/stress-test/multi-gpu/benchmark_eval.sh "$GPU" "$MODEL" "$NICK" &

    next=$(( (next + 1) % ${#GPUS[@]} ))

    # if every GPU already busy, wait for one to finish
    while (( $(jobs -pr | wc -l) >= ${#GPUS[@]} )); do
        sleep 1
    done
done

wait
echo "✔ all models evaluated."
