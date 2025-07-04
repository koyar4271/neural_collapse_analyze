import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet_MNIST(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet_MNIST, self).__init__()
        self.growth_rate = growth_rate
        self.in_planes = 2 * growth_rate

        # MNIST is 1-channel, so in_channels=1
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, self.in_planes, nblocks[0])
        self.in_planes += nblocks[0] * growth_rate
        out_planes = int(self.in_planes * reduction)
        self.trans1 = Transition(self.in_planes, out_planes)
        self.in_planes = out_planes

        self.dense2 = self._make_dense_layers(block, self.in_planes, nblocks[1])
        self.in_planes += nblocks[1] * growth_rate
        out_planes = int(self.in_planes * reduction)
        self.trans2 = Transition(self.in_planes, out_planes)
        self.in_planes = out_planes

        self.dense3 = self._make_dense_layers(block, self.in_planes, nblocks[2])
        self.in_planes += nblocks[2] * growth_rate
        out_planes = int(self.in_planes * reduction)
        self.trans3 = Transition(self.in_planes, out_planes)
        self.in_planes = out_planes

        self.dense4 = self._make_dense_layers(block, self.in_planes, nblocks[3])
        self.in_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes, num_classes)
        self.feature_dim = self.in_planes


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        # For 28x28 images, we might not need all layers
        # out = self.trans3(self.dense3(out))
        # out = self.bn(self.dense4(out))
        out = F.avg_pool2d(F.relu(self.bn(self.dense4(out))), 4)
        
        features = out.view(out.size(0), -1)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        with torch.no_grad():
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
            out = self.trans2(self.dense2(out))
            out = F.avg_pool2d(F.relu(self.bn(self.dense4(out))), 4)
            features = out.view(out.size(0), -1)
        return features


def DenseNet121_MNIST():
    return DenseNet_MNIST(Bottleneck, [6,12,24,16], growth_rate=12)

def DenseNet40_MNIST():
    return DenseNet_MNIST(Bottleneck, [6,6,6,6], growth_rate=12)