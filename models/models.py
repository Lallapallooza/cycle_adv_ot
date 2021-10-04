import torch.nn as nn
from torchvision import models

from models.layers import Identity


class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = models.resnet18(True)
        self.backbone.fc = Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes)
        )

    def forward_seq(self, x):
        features = self.backbone(x)
        return features

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def resnet18(n_classes: int) -> ResNet18:
    return ResNet18(n_classes)
