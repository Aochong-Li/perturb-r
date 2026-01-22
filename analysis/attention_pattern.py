import os
import pickle
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import argparse

BUCKET_NAMES = ['Problem', 'Distractor', 'Response']
FEATURES = ['overall_avg_scores', 'first_128_avg_scores', 'first_256_avg_scores',
            'first_512_avg_scores', 'first_1024_avg_scores']

def load_and_flatten(file_path, feature='overall_avg_scores'):
    """
    Loads the pickle file (list of dicts), converts to DataFrame, 
    and stacks the attention tensors.
    """
    print(f"Loading data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['model_is_correct'])
    
    # Stack features: (N_samples, Layers * Heads * 3)
    features = np.stack(df[feature].values)
        
    labels = df['model_is_correct'].astype(int).values
    n_layers = data[0]['n_layers']
    n_heads = data[0]['n_heads']
    
    return df, features, labels, (n_layers, n_heads)

def analyze_features(args):
    input_path = os.path.join(args.input_dir, f'{args.nickname}.pickle')
    df, X, y, (n_layers, n_heads) = load_and_flatten(input_path, args.feature)
    
    X_correct = X[y == 1]
    X_incorrect = X[y == 0]
    
    print(f"Analyzing {len(y)} samples ({len(X_correct)} Correct, {len(X_incorrect)} Incorrect)")
    print(f"Feature shape: {X.shape} (Layers*Heads, 3_Buckets)")

    # Extract feature name (before _avg_scores)
    feature_name = args.feature.replace('_avg_scores', '')

    # Initialize JSON storage
    json_results = {
        'nickname': args.nickname,
        'feature_type': args.feature,
        'n_samples': int(len(y)),
        'n_correct': int(len(X_correct)),
        'n_incorrect': int(len(X_incorrect)),
        'top_discriminatory_heads': [],
        'lasso_metrics': {},
        'stable_features': []
    }

    # --- TOOL 1: Difference of Means ---
    diff_mean = X_correct.mean(axis=0) - X_incorrect.mean(axis=0)
    
    # --- TOOL 2: Cohen's d (Effect Size) ---
    pooled_std = np.sqrt((X_correct.var(axis=0) + X_incorrect.var(axis=0)) / 2)
    cohens_d = diff_mean / (pooled_std + 1e-9)

    # --- VISUALIZATION ---
    print("Generating Effect Size Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Reshape for visualization: (Layers, Heads, 3)
    cohens_d_reshaped = cohens_d.reshape(n_layers, n_heads, 3)
    
    for i, bucket in enumerate(BUCKET_NAMES):
        heatmap_data = cohens_d_reshaped[:, :, i]
        
        sns.heatmap(heatmap_data, ax=axes[i], center=0, cmap="coolwarm", cbar=True)
        axes[i].set_title(f"Effect Size (Correct > Incorrect)\nAttention to '{bucket}'")
        axes[i].set_xlabel("Head Index")
        axes[i].set_ylabel("Layer Index")
        
        # Identify top discriminatory heads
        flat_indices = np.argsort(np.abs(heatmap_data).ravel())[-3:]
        top_indices = np.unravel_index(flat_indices, heatmap_data.shape)
        
        print(f"\nTop discriminatory heads for '{bucket}' (Layer, Head):")
        for l, h in zip(top_indices[0], top_indices[1]):
             val = heatmap_data[l, h]
             direction = "Correct" if val > 0 else "Incorrect"
             print(f"  L{l}H{h}: d={val:.3f} (favors {direction})")

             # Capture for JSON
             json_results['top_discriminatory_heads'].append({
                 'bucket': bucket,
                 'layer': int(l),
                 'head': int(h),
                 'cohens_d': float(val),
                 'favors': direction
             })

    plt.suptitle(f"Attention Distribution Differences: {args.nickname}", fontsize=16)
    plt.tight_layout()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f'{args.nickname}_{feature_name}_cohens_d.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    # --- TOOL 3: Lasso with Train/Test Split ---
    print("\n--- Lasso Regression ---")

    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(penalty='l1', C=args.C, solver='liblinear',
                             class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=args.cv_folds, scoring='accuracy')

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"CV accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    cm_test = confusion_matrix(y_test, clf.predict(X_test))
    print(f"\nConfusion Matrix (Test):\n{cm_test}")

    # Capture Lasso Metrics for JSON
    json_results['lasso_metrics'] = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'confusion_matrix': cm_test.tolist()
    }

    # Feature stability across CV folds
    coeffs = clf.coef_.reshape(n_layers, n_heads, 3)
    feature_counts = np.zeros((n_layers, n_heads, 3))

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    for train_idx, _ in skf.split(X_scaled, y):
        fold_clf = LogisticRegression(penalty='l1', C=args.C, solver='liblinear',
                                       class_weight='balanced', random_state=42)
        fold_clf.fit(X_scaled[train_idx], y[train_idx])
        fold_coeffs = fold_clf.coef_.reshape(n_layers, n_heads, 3)
        feature_counts += (np.abs(fold_coeffs) > 1e-4).astype(int)

    stability_threshold = int(0.8 * args.cv_folds)
    stable_features = feature_counts >= stability_threshold

    print(f"\nStable features (≥{stability_threshold}/{args.cv_folds} folds): {stable_features.sum()}")

    if stable_features.sum() > 0:
        stable_indices = np.argwhere(stable_features)
        for layer, head, bucket_idx in stable_indices:
            coef = coeffs[layer, head, bucket_idx]
            direction = "Correct" if coef > 0 else "Incorrect"
            bucket_name = BUCKET_NAMES[bucket_idx]
            print(f"  L{layer}H{head} {bucket_name:10s} {coef:+.3f} → {direction}")
            
            # Capture Stable Features for JSON
            json_results['stable_features'].append({
                'layer': int(layer),
                'head': int(head),
                'bucket': bucket_name,
                'coefficient': float(coef),
                'direction': direction
            })

    # Save standard pickle results
    results = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'coefficients': coeffs.tolist(),
        'feature_stability': feature_counts.tolist(),
        'confusion_matrix': cm_test.tolist(),
    }

    results_path = os.path.join(args.output_dir, f'{args.nickname}_{feature_name}_lasso_results.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nPickle results saved to {results_path}")

    # Save JSON results (Human readable / Terminal summary)
    json_path = os.path.join(args.output_dir, f'{args.nickname}_{feature_name}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"JSON summary saved to {json_path}")

if __name__ == "__main__":
    """
    python analysis/attention_pattern.py \
        --input_dir ./results/allmath/inject_distractor/attention_scores \
        --output_dir ./results/allmath/inject_distractor/attention_pattern
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nickname', type=str, required=False)
    parser.add_argument('--feature', type=str, required=False)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--C', type=float, default=0.1)
    parser.add_argument('--cv_folds', type=int, default=5)
    args = parser.parse_args()

    # Get all available nicknames and features
    all_nicknames = []
    if os.path.exists(args.input_dir):
        for file in os.listdir(args.input_dir):
            if file.endswith('.pickle'):
                nickname = file.replace('.pickle', '')
                all_nicknames.append(nickname)
    
    all_features = FEATURES
    
    # Determine which nicknames and features to process
    nicknames_to_process = [args.nickname] if args.nickname else all_nicknames
    features_to_process = [args.feature] if args.feature else all_features
    
    # Loop over all combinations
    for nickname in nicknames_to_process:
        for feature in features_to_process:
            print(f"\n{'='*80}")
            print(f"Processing: {nickname} - {feature}")
            print(f"{'='*80}")
            
            # Create a copy of args with the specific nickname and feature
            current_args = argparse.Namespace(**vars(args))
            current_args.nickname = nickname
            current_args.feature = feature
            
            try:
                analyze_features(current_args)
            except Exception as e:
                print(f"Error processing {nickname} - {feature}: {e}")
                continue