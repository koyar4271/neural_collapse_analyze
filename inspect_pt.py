'''
python3 inspect_pt.py --path ./features/epoch_001.pt
'''
import torch
import argparse

def summarize_tensor(name, tensor):
    print(f"{name}:")
    print(f"  Type: {type(tensor)}")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    if tensor.numel() < 10:
        print(f"  Values: {tensor}")
    else:
        print(f"  Sample values: {tensor.flatten()[:5]} ...")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .pt file")
    args = parser.parse_args()

    data = torch.load(args.path, map_location='cpu', weights_only=True)
    print(f"Loaded: {args.path}")
    print(f"Keys: {list(data.keys())}\n")

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            summarize_tensor(key, value)
        else:
            print(f"{key}: (non-tensor, type = {type(value)})\n")

if __name__ == "__main__":
    main()
