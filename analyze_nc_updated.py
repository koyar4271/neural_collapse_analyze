# analyze_nc_updated.py
"""
python analyze_nc_updated.py --feature_dir ./features/cifar10/cifar10_05 --csv_out ./results/ResNet/result_cifar10_05.csv
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

EPS = 1e-12

def ensure_W_DK(W: torch.Tensor, d_expected: int = None, k_expected: int = None) -> torch.Tensor:
    """
    W を R^{d×K} 形状に整える。入力は (K,d) or (d,K) を想定。
    """
    if W.ndim != 2:
        raise ValueError("classifier_weight must be 2D.")
    Wr, Wc = W.shape
    # 期待次元が与えられていればそれを優先して判定
    if d_expected is not None and k_expected is not None:
        if (Wr == d_expected and Wc == k_expected):
            return W
        if (Wr == k_expected and Wc == d_expected):
            return W.t()
    # ヒューリスティック：列数がクラス数になりがち（K は d より小さいことが多い）
    if Wr <= Wc:
        return W.t()  # (K,d) -> (d,K)
    return W         # (d,K)

def build_class_means(H: torch.Tensor, labels: torch.Tensor, K: int):
    """
    H: (N,d), labels: (N,), K: クラス数
    戻り値:
      class_means: (d,K), global_mean: (d,)
      idx_of_class: ユニークラベルの昇順（W の列と同順を仮定）
      counts: (K,)
    """
    # ユニークラベル（昇順）
    uniq = torch.unique(labels.cpu()).tolist()
    # 期待: uniq が [0,1,...,K-1] で長さ K
    if len(uniq) != K or any(int(u) != u for u in uniq) or sorted(uniq) != list(range(K)):
        # 必要ならここで W 列とラベルのマッピングを外部から渡す設計に拡張可
        # 現状は W の列順 = 0..K-1 を前提に、ラベルを強制的に再マッピング
        mapping = {old: new for new, old in enumerate(sorted(uniq))}
        labels = labels.clone()
        for old, new in mapping.items():
            labels[labels == old] = new

    d = H.shape[1]
    class_means = H.new_zeros(d, K)
    counts = torch.zeros(K, dtype=torch.long, device=H.device)
    for k in range(K):
        mask = (labels == k)
        if mask.any():
            class_means[:, k] = H[mask].mean(dim=0)
            counts[k] = int(mask.sum().item())
    global_mean = H.mean(dim=0)
    return class_means, global_mean, labels, counts

def compute_nc1_from_svd(Sigma_W: torch.Tensor, H_bar: torch.Tensor) -> torch.Tensor:
    """
    NC1 = (1/K) * trace(Σ_W Σ_B^†),
      Σ_B = (1/K) H_bar H_bar^T
    H_bar: (d,K) = [\bar h_k - h_G] の列集合
    安定化のため、H_bar の SVD から Σ_B^† を構成。
    """
    d, K = H_bar.shape
    # H_bar = U S V^T (full_matrices=False) の rank は ≤ K-1
    U, S, _ = torch.linalg.svd(H_bar, full_matrices=False)
    # NC の理想では rank = K-1。S の最後の1つが 0 になることが多い
    # Σ_B = (1/K) U diag(S^2) U^T  ->  Σ_B^† = K * U_r diag(1/S_r^2) U_r^T
    # ここで r は S > 0 の成分数（通常 K-1）
    pos = (S > 1e-10)
    if pos.sum() == 0:
        return torch.tensor(float('nan'), dtype=Sigma_W.dtype, device=Sigma_W.device)
    U_r = U[:, pos]                # (d,r)
    S_r = S[pos]                   # (r,)
    inv_S2 = 1.0 / (S_r * S_r)     # (r,)
    # trace(Σ_W Σ_B^†) = K * trace(U_r^T Σ_W U_r diag(1/S_r^2))
    M = U_r.transpose(0,1) @ Sigma_W @ U_r     # (r,r)
    val = K * torch.sum(torch.diag(M) * inv_S2)
    return (val / K)               # ここで (1/K) を掛ける → 結果的に just `val/K`
    # 実質、上3行で K を掛け、最後に 1/K を掛けているので「係数抜け」が無い状態

def compute_metrics_from_paper(features: torch.Tensor, labels: torch.Tensor, classifier_weight: torch.Tensor):
    """
    「Cross Entropy versus Label Smoothing」式(4)(5)(6)に基づく NC1, NC2, NC3。
    NC1: (1/K) * trace(Σ_W Σ_B^†)
    NC2: || (W^T H̄)/||W^T H̄||_F - (1/sqrt(K-1))(I - 11^T/K) ||_F
    NC3: || W/||W||_F - H̄/||H̄||_F ||_F
    （H̄ はクラス平均をグローバル平均でセンタリングした d×K 行列）
    """
    # --- 型・精度統一 ---
    device = features.device
    H = features.to(dtype=torch.float64)
    labs = labels.to(dtype=torch.long)
    W_in = classifier_weight.to(dtype=torch.float64)

    N, d = H.shape
    # W を d×K に整形
    W = ensure_W_DK(W_in, d_expected=d)
    dW, K = W.shape

    if dW != d:
        raise ValueError(f"Incompatible dims: features d={d}, but W has d={dW}.")

    # --- クラス平均とグローバル平均 ---
    class_means, global_mean, labs_fixed, counts = build_class_means(H, labs, K)
    # H̄（d×K）
    H_bar = class_means - global_mean.unsqueeze(1)

    # --- Σ_W（1/N スケーリング、式(4)と整合）---
    Sigma_W = torch.zeros((d, d), dtype=torch.float64, device=device)
    for k in range(K):
        mask = (labs_fixed == k)
        if mask.any():
            Xc = H[mask] - class_means[:, k].unsqueeze(0)  # (n_k, d)
            Sigma_W += Xc.t() @ Xc
    Sigma_W /= max(N, 1)

    # --- NC1（式(4)）---
    NC1 = compute_nc1_from_svd(Sigma_W, H_bar)

    # --- NC3（式(6)）---
    W_normF = torch.linalg.norm(W, ord='fro')
    H_normF = torch.linalg.norm(H_bar, ord='fro')
    if (W_normF > EPS) and (H_normF > EPS):
        NC3 = torch.linalg.norm(W / W_normF - H_bar / H_normF, ord='fro')
    else:
        NC3 = torch.tensor(float('nan'), dtype=torch.float64, device=device)

    # --- NC2（式(5)）---
    M_ETF = (torch.eye(K, dtype=torch.float64, device=device) - torch.ones((K,K), dtype=torch.float64, device=device)/K) / np.sqrt(max(K-1,1))
    WH = W.t() @ H_bar   # (K,K)
    WH_normF = torch.linalg.norm(WH, ord='fro')
    if WH_normF > EPS:
        NC2 = torch.linalg.norm(WH / WH_normF - M_ETF, ord='fro')
    else:
        NC2 = torch.tensor(float('nan'), dtype=torch.float64, device=device)

    return {
        'NC1': float(NC1.item()),
        'NC2': float(NC2.item()),
        'NC3': float(NC3.item()),
        'H_norm': float(H_normF.item()),
        'W_norm': float(W_normF.item())
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze NC metrics per epoch (paper-true definitions).")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--csv_out", type=str, required=True)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        print(f"Error: Directory not found at {feature_dir}")
        return

    # "epoch_XX.pt" の XX を数値でソート
    def _key(p: Path):
        try:
            return int(p.stem.split('_')[1])
        except Exception:
            return 1 << 30

    epoch_files = sorted(feature_dir.glob("epoch_*.pt"), key=_key)
    if not epoch_files:
        print(f"Error: No 'epoch_*.pt' files found in {feature_dir}")
        return

    print(f"Found {len(epoch_files)} epoch files. Analyzing...")

    rows = []
    for ep_path in tqdm(epoch_files, desc="Analyzing epochs"):
        try:
            data = torch.load(ep_path, map_location='cpu')
            # 必須キー
            needed = ['features', 'labels', 'classifier_weight']
            if not all(k in data for k in needed):
                print(f"Warning: {ep_path.name} skipped (missing one of {needed}).")
                continue

            feats = data['features']
            labs  = data['labels']
            W     = data['classifier_weight']

            # NC指標
            metrics = compute_metrics_from_paper(feats, labs, W)

            # 付随情報（任意）
            row = {
                'Epoch': ep_path.stem,
                'TestAcc': data.get('test_accuracy', np.nan),
                'TrainAcc': data.get('train_accuracy', np.nan),
                'TrainLoss': data.get('train_loss', np.nan),
                **metrics
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing {ep_path.name}: {e}")

    if not rows:
        print("No results were generated. Please check the input files.")
        return

    df = pd.DataFrame(rows)
    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")

    print(f"\nAnalysis complete. Results saved to {args.csv_out}")
    print("\nCSV Head:")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
