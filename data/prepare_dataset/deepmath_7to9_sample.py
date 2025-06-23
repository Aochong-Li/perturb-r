#!/usr/bin/env python3
"""
Script to prepare DeepMath dataset with difficulty levels 7, 8, 9.
Samples 300 data points from the test split and saves to ./data/deepmath_7to9
"""

import os
from datasets import load_dataset, DatasetDict
import argparse
import shutil

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Prepare DeepMath 7-9 difficulty dataset')
    parser.add_argument('--output_dir', default='./data/deepmath_7to9', 
                       help='Output directory to save the dataset')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='Number of samples to extract')
    args = parser.parse_args()
    
    try:
        # Load the test split from the dataset
        dataset = load_dataset("aochongoliverli/DeepMath-103K-split", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Filter for difficulty levels 7, 8, 9
    filtered_dataset = dataset.filter(lambda x: x['difficulty'] in [7, 8, 9])
    
    if len(filtered_dataset) == 0:
        print("No samples found with difficulty 7, 8, or 9!")
        return
    
    # Sample the specified number of data points
    if len(filtered_dataset) > args.sample_size:
        sampled_dataset = filtered_dataset.shuffle(seed=42).select(range(args.sample_size))
    else:
        sampled_dataset = filtered_dataset
    
    # Rename columns as requested: question -> problem, final_answer -> solution
    def rename_columns(example):
        if 'question' in example:
            example['problem'] = example.pop('question')
        if 'final_answer' in example:
            example['solution'] = example.pop('final_answer')
        return example
    
    final_dataset = sampled_dataset.map(rename_columns)
    final_dataset = DatasetDict({
        'test': final_dataset
    })
    # Create output directory if it doesn't exist
    create_dir(args.output_dir)
    
    # Save the dataset using save_to_disk
    final_dataset.save_to_disk(args.output_dir)
    
    print(f"Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
