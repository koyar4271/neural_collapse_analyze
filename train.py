import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model import PapyanCNN
from pathlib import Path
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_epochs = 2
lr = 0.05
weight_decay = 5e-4
momentum = 0.9
lr_milestones = [int(num_epochs * 1 / 3), int(num_epochs * 2 / 3)]

transform = transforms.ToTensor() # include normalization
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

targets = train_dataset.targets.numpy()
indices = []
for c in range(10):
    idx = np.where(targets == c)[0][:5000]  # 5000 images each class
    indices.extend(idx)
subset = torch.utils.data.Subset(train_dataset, indices)
train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
# initialize dataloader

model = PapyanCNN().to(device)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# settings of optimization SGD method
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
# how to decrease lr, lr *= 0.1 at lr_milestones

save_dir = Path('./features') # directory to save features
save_dir.mkdir(exist_ok=True)

for epoch in range(1, num_epochs + 1):
    model.train() # switch to training mode
    total_loss = 0
    correct = 0
    total = 0

    print(f"Epoch {epoch}/{num_epochs}")
    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad() # reset
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    scheduler.step()

    acc = correct / total * 100
    print(f"→ Loss={total_loss/total:.4f}, Acc={acc:.2f}%")

    # 特徴とラベルを保存
    model.eval() # switch to evaluation mode
    all_feats = []
    all_labs = []
    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="Extracting", leave=False):
            x = x.to(device)
            feats = model.extract_features(x)
            all_feats.append(feats.cpu())
            all_labs.append(y)

    feats = torch.cat(all_feats)
    labs = torch.cat(all_labs)
    torch.save({'features': feats, 'labels': labs,
                'classifier_weight': model.classifier.weight.detach().cpu()},
                save_dir / f'epoch_{epoch:03d}.pt')
