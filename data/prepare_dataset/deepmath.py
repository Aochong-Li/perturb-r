import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse
import shutil

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def process_fn(example):
    question = example['question']
    solution = example['final_answer']

    return {
        'problem': question,
        'solution': solution,
        'task': 'math'
        }

def main():
    """
    python ./data/prepare_dataset/deepmath.py --output_dir ./data/deepmath --min_level 1 --max_level 9
    """
    parser = argparse.ArgumentParser(description='Prepare Countdown dataset')
    parser.add_argument('--output_dir', default='./data/countdown', 
                       help='Output directory to save the dataset')
    parser.add_argument('--min_level', type=int, required=True,
                       help='Minimum level to prepare')
    parser.add_argument('--max_level', type=int, required=True,
                       help='Maximum level to prepare')
    args = parser.parse_args()
    
    test_dataset = load_dataset("aochongoliverli/DeepMath-103K")["test"]
    test_dataset = test_dataset.filter(lambda x: x['difficulty'] >= args.min_level and x['difficulty'] <= args.max_level)
    test_dataset = test_dataset.map(process_fn, remove_columns=['question', 'final_answer', 'r1_solution_1', 'r1_solution_2', 'r1_solution_3'])
    
    dataset = DatasetDict({
        'test': test_dataset
    })
    create_dir(args.output_dir)

    dataset.save_to_disk(args.output_dir)
    
    print(f"Dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
