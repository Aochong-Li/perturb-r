import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse
import shutil

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def process_fn(example, level):
    numbers = example['nums']
    target = example['target']
    
    problem = f"Please reason and answer the following question. Using the numbers {numbers}, create an equation that equals {target}. You can only use basic arithmetic operations (+, -, *, /) in the expression and each number should be used exactly once. You should report the answer equation expression in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    
    return {
        'problem': problem,
        'level': level,
        'task': 'countdown'
        }

def main():
    """
    python ./data/prepare_dataset/countdown.py --output_dir ./data/countdown_train_stage2_level5_35K --levels 5 --sample_size_per_level 35000 --skip_first_n 110000 --split train
    python ./data/prepare_dataset/countdown.py --output_dir ./data/countdown --levels 4 5 6 7 9 11 13 --sample_size_per_level 200
    python ./data/prepare_dataset/countdown.py --output_dir /share/goyal/lio/reasoning/data/countdown/sft/level3-4 --levels 3 4 --sample_size_per_level 5000
    """
    parser = argparse.ArgumentParser(description='Prepare Countdown dataset')
    parser.add_argument('--output_dir', default='./data/countdown', 
                       help='Output directory to save the dataset')
    parser.add_argument('--skip_first_n', type=int, default=0, required=False, help="For each level, skip the first n samples")
    parser.add_argument('--sample_size_per_level', type=int, default=100,
                       help='Number of samples to extract per level')
    parser.add_argument('--levels', type=int, nargs="+", required=True,
                       help='Levels to prepare')
    parser.add_argument('--split', type=str, default='test', required=False, help="Split to prepare")
    args = parser.parse_args()
    
    final_dataset = []
    for level in args.levels:
        dataset = load_dataset(f"aochongoliverli/countdown_level_{level}", split=args.split)
        sampled_dataset = dataset.select(range(args.skip_first_n, args.skip_first_n + args.sample_size_per_level))
        sampled_dataset = sampled_dataset.map(lambda example: process_fn(example, level)).rename_columns({"solution": "example_solution"})
        final_dataset.append(sampled_dataset)

    final_dataset = DatasetDict({
        'test': concatenate_datasets(final_dataset)
    })

    
    # Create output directory if it doesn't exist
    create_dir(args.output_dir)
    # Save the dataset using save_to_disk
    final_dataset.save_to_disk(args.output_dir)
    
    print(f"Dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
