'''
python comp.py --files results/result_mnist_soft.csv results/mnist/result_mnist_005.csv results/mnist_distill.csv --labels "label-smoothing" "one-hot" "distill"
'''
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_comparison(files, labels):
    """
    Plots a comparison of specified metrics from multiple CSV files.

    Args:
        files (list): A list of paths to the CSV files.
        labels (list): A list of labels for each file.
    """
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Standardize the 'Epoch' column to be numeric
            if 'Epoch' in df.columns and isinstance(df['Epoch'].iloc[0], str):
                df['Epoch'] = df['Epoch'].str.split('_').str[-1].astype(int)
            df = df.sort_values(by='Epoch')
            dataframes.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found at {file}, skipping.")
        except (KeyError, AttributeError):
            print(f"Warning: 'Epoch' column in {file} is not in the expected format, skipping.")

    if not dataframes:
        print("No valid data to plot.")
        return

    # Use the columns from the first valid dataframe as the metrics to plot
    metrics = [col for col in dataframes[0].columns if col != "Epoch"]
    
    # Define markers to cycle through for each line
    markers = ['o', 's', 'v', '^', '<', '>', 'D', 'p']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for i, (df, label) in enumerate(zip(dataframes, labels)):
            if metric in df.columns:
                plt.plot(df["Epoch"], df[metric], marker=markers[i % len(markers)], label=label)

        plt.title(metric, fontsize=16)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path('./plots/comparisons')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'{metric}_comparison.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare metrics from multiple experiments.")
    parser.add_argument('--files', nargs='+', required=True, help='A list of paths to the input CSV files.')
    parser.add_argument('--labels', nargs='+', required=True, help='A list of labels for each corresponding file.')
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        print("Error: The number of files must match the number of labels.")
    else:
        plot_comparison(args.files, args.labels)