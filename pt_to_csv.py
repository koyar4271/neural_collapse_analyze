'''
python3 pt_to_csv.py --path ./features/epoch_001.pt --out_dir ./csv_epoch001
'''
import torch
import argparse
import os
import numpy as np
import pandas as pd

def save_features_with_labels(features, labels, output_path):
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy().reshape(-1, 1)
    combined = np.concatenate([labels_np, features_np], axis=1)

    columns = ['label'] + [f'feat_{i}' for i in range(features_np.shape[1])]
    df = pd.DataFrame(combined, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Saved labeled features to {output_path}")

def save_tensor_to_csv(tensor, filename):
    array = tensor.cpu().detach().numpy()
    df = pd.DataFrame(array)
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--out_dir', type=str, default='csv_output', help='Directory to save CSV files')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = torch.load(args.path, map_location='cpu', weights_only=True)

    if 'features' in data and 'labels' in data:
        save_features_with_labels(
            data['features'], data['labels'],
            os.path.join(args.out_dir, 'features_with_labels.csv')
        )

    for key, tensor in data.items():
        if key in ['features', 'labels']:
            continue  # already handled
        if isinstance(tensor, torch.Tensor):
            save_tensor_to_csv(tensor, os.path.join(args.out_dir, f'{key}.csv'))
        else:
            print(f"Skipping non-tensor key: {key} (type={type(tensor)})")

if __name__ == "__main__":
    main()
