'''
python plot.py --csv_path result_cifar10.csv --columns TestAcc NC1 NC3 --save_dir plots/result_cifar10
'''
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the input CSV file')
    parser.add_argument('--columns', type=str, nargs='+', required=True,
                        help='List of column names to plot')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the output plot images')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["Epoch"] = df["Epoch"].str.extract(r'epoch_(\d+)').astype(int)

    for col in args.columns:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in CSV.")
            continue

        plt.figure()
        plt.plot(df["Epoch"], df[col], marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{col} over Epochs")
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"{col}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()