import torch
import numpy as np
from pathlib import Path
import argparse

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_nc_metrics(features, labels, classifier_weight=None):
    '''
    Compute NC1, NC2, NC3 metrics from feature and label tensors.
    features: (N, D)
    labels: (N,)
    classifier_weight: (C, D) if available
    '''
    features = features.numpy()
    labels = labels.numpy()
    classes = np.unique(labels)
    C = len(classes)
    D = features.shape[1]

    # Class means
    class_means = np.zeros((C, D))
    for c in classes:
        class_means[c] = features[labels == c].mean(axis=0)
    global_mean = features.mean(axis=0)

    # NC1: within-class variance / total variance
    sigma_w = 0
    sigma_t = ((features - global_mean) ** 2).sum()
    for c in classes:
        class_feats = features[labels == c]
        sigma_w += ((class_feats - class_means[c]) ** 2).sum()
    nc1 = sigma_w / sigma_t

    # NC2: pairwise cosine similarity between class means
    sims = []
    for i in range(C):
        for j in range(i+1, C):
            sims.append(cosine_similarity(class_means[i], class_means[j]))
    nc2 = np.std(sims)

    # NC3: alignment between classifier weights and class means (if available)
    nc3 = None
    if classifier_weight is not None:
        W = classifier_weight.cpu().detach().numpy()
        norm_means = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
        norm_W = W / np.linalg.norm(W, axis=1, keepdims=True)
        nc3 = np.mean([cosine_similarity(norm_means[c], norm_W[c]) for c in range(C)])

    return nc1, nc2, nc3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="./features")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--csv_out", type=str, default="nc_results.csv")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    epochs = sorted(feature_dir.glob("epoch_*.pt"))
    print(f"Found {len(epochs)} epochs")

    # モデル重みの取得（任意）
    classifier_weight = None
    if args.model_path:
        model_data = torch.load(args.model_path, map_location='cpu')
        if isinstance(model_data, dict) and 'classifier.weight' in model_data:
            classifier_weight = model_data['classifier.weight']
        else:
            for k in model_data:
                if 'classifier.weight' in k:
                    classifier_weight = model_data[k]
                    break

    print("Epoch, NC1, NC2, NC3")
    with open(args.csv_out, "w") as f:
        f.write("Epoch,NC1,NC2,NC3\n")
        for ep in epochs:
            data = torch.load(ep, map_location='cpu', weights_only=True)
            feats = data['features']
            labs = data['labels']

            # .pt内に分類器が含まれていれば優先して使う
            weight = data.get('classifier_weight', classifier_weight)
            nc1, nc2, nc3 = compute_nc_metrics(feats, labs, weight)
            print(f"{ep.stem},{nc1:.4f},{nc2:.4f},{nc3 if nc3 is not None else 'N/A'}")
            f.write(f"{ep.stem},{nc1:.4f},{nc2:.4f},{nc3 if nc3 is not None else 'N/A'}\n")

if __name__ == "__main__":
    main()
