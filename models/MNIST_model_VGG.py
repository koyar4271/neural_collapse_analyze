import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
}

class VGG_MNIST(nn.Module):
    def __init__(self, vgg_name='VGG11', num_classes=10):
        super(VGG_MNIST, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.feature_dim = 512

    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        logits = self.classifier(features_flat)
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            features = self.features(x)
            features_flat = features.view(features.size(0), -1)
        return features_flat

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)