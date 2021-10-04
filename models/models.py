import torch.nn as nn
from torchvision import models

from models.layers import Identity


class ResNet(nn.Module):
    def __init__(self, backbone, n_classes, emb_size, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.emb_size = emb_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.fc = Identity()
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, n_classes)
        )

    def forward_seq(self, x):
        features = self.backbone(x)
        return features

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def resnet18(n_classes: int, freeze_backbone=True) -> ResNet:
    return ResNet(models.resnet18(True), n_classes, 512, freeze_backbone)

def resnet50(n_classes: int, freeze_backbone=True) -> ResNet:
    return ResNet(models.resnet50(True), n_classes, 2048, freeze_backbone)
