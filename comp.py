'''
python comp.py results/result_mnist_soft.csv results/mnist/result_mnist_005.csv "label-smoothing" "one-hot"
'''
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_comparison(file1, file2, label1, label2):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    epochs1 = [int(e.split('_')[1]) for e in df1["Epoch"]]
    epochs2 = [int(e.split('_')[1]) for e in df2["Epoch"]]

    metrics = [col for col in df1.columns if col != "Epoch"]

    for metric in metrics:
        plt.figure()
        plt.plot(epochs1, df1[metric], marker='o', label=label1)
        plt.plot(epochs2, df2[metric], marker='s', label=label2)
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{metric}.png')
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python comp.py file1.csv file2.csv label1 label2")
    else:
        file1, file2, label1, label2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        plot_comparison(file1, file2, label1, label2)
