import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from data import get_dataloaders
from config import get_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

DATASET = 'mnist'
config = get_config(DATASET)

batch_size = config['batch_size']
num_epochs = config['num_epochs']
lr = config['lr']
weight_decay = config['weight_decay']
momentum =config['momentum']
lr_milestones = config['lr_milestones']

train_loader, test_loader = get_dataloaders(dataset_name=DATASET, batch_size=batch_size)

if DATASET == 'mnist':
    from models.MNIST_model import PapyanCNN
    model = PapyanCNN().to(device)
elif DATASET == 'cifar10':
    from models.CIFAR10_model import PapyanResNet18
    model = PapyanResNet18().to(device)
else:
    raise ValueError("Invalid dataset")

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

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct_test += (preds == y).sum().item()
            total_test += x.size(0)
    test_acc = correct_test / total_test * 100
    print(f"→ Test Acc={test_acc:.2f}%")

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
    torch.save({'features': feats,
                'labels': labs,
                'classifier_weight': model.classifier.weight.detach().cpu(),
                'train_loss': total_loss / total,
                'train_accuracy': acc,
                'test_accuracy': test_acc
                },
                save_dir / f'epoch_{epoch:03d}.pt')
