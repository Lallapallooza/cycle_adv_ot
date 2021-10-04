import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""

    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


def basic_resnet18_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def hard_augment_resnet18_transform(n_channels):
    if n_channels == 3:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.randn((224,224)) * 0.005),
            transforms.Lambda(lambda t: t.clamp(0,1))
        ])
    elif n_channels == 1:
        return transforms.Compose([
            GrayscaleToRgb(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.randn((224,224)) * 0.005),
            transforms.Lambda(lambda t: t.clamp(0,1))
        ])
    else:
        raise RuntimeError("Only RGB and Grayscale images are supported")

def small_augment_resnet18_transform(n_channels):
    if n_channels == 3:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
    elif n_channels == 1:
        return transforms.Compose([
            GrayscaleToRgb(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
    else:
        raise RuntimeError("Only RGB and Grayscale images are supported")
