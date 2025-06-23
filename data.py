# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(dataset_name: str, batch_size=128, num_workers=2, samples_per_class=5000):
    if dataset_name.lower() == 'mnist':
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        # サブサンプリング（クラスごとに5000枚）
        targets = train_dataset.targets.numpy()
        indices = []
        for c in range(10):
            idx = np.where(targets == c)[0][:samples_per_class]
            indices.extend(idx)
        train_dataset = Subset(train_dataset, indices)

    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616])
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

        targets = np.array(train_dataset.targets)
        indices = []
        for c in range(10):
            idx = np.where(targets == c)[0][:samples_per_class]
            indices.extend(idx)
        train_dataset = Subset(train_dataset, indices)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        targets = np.array(train_dataset.targets)
        indices = []
        for c in range(100):
            idx = np.where(targets == c)[0][:samples_per_class]
            indices.extend(idx)
        train_dataset = Subset(train_dataset, indices)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # ←追加！

    return train_loader, test_loader, eval_loader
