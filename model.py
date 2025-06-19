import torch
import torch.nn as nn

class PapyanCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc1(x)
        return self.classifier(x)

    def extract_features(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.fc1(x)
        return x
