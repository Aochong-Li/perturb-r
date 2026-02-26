import os
import pickle
import numpy as np
import pandas as pd
import argparse

BUCKET_NAMES = ['Problem', 'Distractor', 'Response']

FEATURES = ['overall_avg_scores', 'first_128_avg_scores', 'first_256_avg_scores',
            'first_512_avg_scores', 'first_1024_avg_scores']

def load_data(input_dir, nickname):
    """Load pickle file and return features and labels."""
    file_path = os.path.join(input_dir, f'{nickname}.pickle')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    df = df.dropna(subset=['model_is_correct'])

    return df

def compute_means(df, feature):
    """Compute mean attention for correct vs incorrect."""
    labels = df['model_is_correct'].astype(int).values
    features = np.stack(df[feature].values)  # Shape: (N, n_layers*n_heads, 3)

    # Average over all layers and heads
    features_avg = features.mean(axis=1)  # Shape: (N, 3)

    correct_mean = features_avg[labels == 1].mean(axis=0)
    incorrect_mean = features_avg[labels == 0].mean(axis=0)

    return correct_mean, incorrect_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    # Load data
    df1 = load_data(args.input_dir, args.model1)
    df2 = load_data(args.input_dir, args.model2)

    print(f"\nModel 1: {args.model1} ({len(df1)} samples)")
    print(f"Model 2: {args.model2} ({len(df2)} samples)")
    print("="*80)

    # Compare each feature
    for feature in FEATURES:
        print(f"\n{feature.replace('_avg_scores', '').replace('_', ' ').upper()}")
        print("-"*80)

        for model_name, df in [(args.model1, df1), (args.model2, df2)]:
            correct_mean, incorrect_mean = compute_means(df, feature)

            print(f"\n{model_name}:")
            print(f"  Correct:   {' '.join([f'{BUCKET_NAMES[i]}={correct_mean[i]:.3f}' for i in range(3)])}")
            print(f"  Incorrect: {' '.join([f'{BUCKET_NAMES[i]}={incorrect_mean[i]:.3f}' for i in range(3)])}")
            print(f"  Diff:      {' '.join([f'{BUCKET_NAMES[i]}={correct_mean[i]-incorrect_mean[i]:+.3f}' for i in range(3)])}")

if __name__ == "__main__":
    main()
