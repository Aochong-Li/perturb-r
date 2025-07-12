#!/usr/bin/env bash
set -euo pipefail

# This script now uses a Python-based scheduler for more robust job management.
# Configure your available GPUs by setting the CUDA_VISIBLE_DEVICES environment variable.
# For example:
#
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#
# If not set, the scheduler will attempt to detect all available GPUs using nvidia-smi.

# Get the directory of the script to robustly find the scheduler.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCHEDULER_PY="$SCRIPT_DIR/scheduler.py"

if [ ! -f "$SCHEDULER_PY" ]; then
    echo "Error: Scheduler script not found at $SCHEDULER_PY"
    exit 1
fi

# The python script assumes it's run from the project root (e.g., 'projects/perturb-r')
# so that default paths to configs and scripts are correct.
python3 "$SCHEDULER_PY"

echo "✔ All jobs launched by scheduler."
