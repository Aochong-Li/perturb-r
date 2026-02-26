# Cross-Model Distractor Injection Ablation Study

This directory contains ablation studies for the inject_distractor experiment.

## inject_distractor_cross_model.py

This script performs an ablation study where distractors are sampled from **different models** instead of the same model being tested.

### Key Differences from Original

**Original experiment** (`stress-test/inject_distractor.py`):
- Samples distracting reasoning from the same model
- Uses problems not in the test set as distractors

**Cross-model ablation** (`stress-test/ablation/inject_distractor_cross_model.py`):
- **Directly reuses distractor columns** from 5 randomly selected other models' pickle files
- Creates a pool of all distractor rows from these 5 models
- Randomly pairs each test row with a distractor from this cross-model pool
- Much simpler: no need to recompute reasoning chunks or select specific problems
- Adds a `distractor_model` column to track which model the distractor came from

### Usage

```bash
python stress-test/ablation/inject_distractor_cross_model.py \
    --model_name <model_path> \
    --nick_name <model_nickname> \
    --tokenizer_name <tokenizer_path> \
    --results_dir <path_to_results_dir> \
    --num_distractor_models 5 \
    --granularity 30 \
    --unit 0.2 \
    --overwrite
```

### Arguments

- `--model_name`: Path to the model to test
- `--nick_name`: Nickname of the model (must match a pickle file in `results_dir/inject_distractor/`)
- `--tokenizer_name`: Path to the tokenizer
- `--results_dir`: Directory containing `inject_distractor` subdirectory with pickle files (default: `/share/goyal/lio/reasoning/eval/`)
- `--num_distractor_models`: Number of different models to randomly select as distractor sources (default: 5)
- `--overwrite`: Overwrite existing results

### How It Works

1. Loads the test model's `inject_distractor` pickle file
2. Drops the original distractor-related columns from the test model
3. Randomly selects 5 other models' pickle files
4. Creates a pool of all distractor rows from these 5 models (with their distractor columns intact)
5. Randomly samples from this pool to pair each test row with a cross-model distractor
6. Adds `distractor_model` column to track the source model
7. Assembles prompts and runs evaluation

### Prerequisites

The script requires:
1. Existing results from the original inject_distractor experiment in `{results_dir}/inject_distractor/`
2. At least `num_distractor_models` other model pickle files in the same directory

### Output

Results are saved to: `{results_dir}/inject_distractor_cross_model/{nick_name}.pickle`

The output dataframe includes:
- Original test columns: `problem`, `solution`, `source`, `original_response`, `solve_n`, `reasoning_chunks`, `original_ratio`
- Cross-model distractor info: `distractor_model`, `distractor_ratio`, `distractor_problem`, `distractor_solution`, `distractor_source`, `distractor_reasoning_chunks`, `distractor_solve_n`
- Generated outputs: `prompt`, `post_distraction_response`, `pred`, `ground_truth`

### Example

```bash
# Test Qwen3-1.7B with distractors from 5 other models
python stress-test/ablation/inject_distractor_cross_model.py \
    --model_name /path/to/Qwen3-1.7B \
    --nick_name Qwen3-1.7B \
    --tokenizer_name /path/to/Qwen3-1.7B \
    --results_dir /share/goyal/lio/reasoning/eval/ \
    --tensor_parallel_size 1 \
    --overwrite
```

### Analysis

To analyze the results, compare:
1. Original inject_distractor results (distractors from same model)
2. Cross-model results (distractors from different models)

This helps understand whether reasoning models are more susceptible to distractors from:
- Their own reasoning style
- Different models' reasoning patterns
