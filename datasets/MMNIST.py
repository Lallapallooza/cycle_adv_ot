
from datasets.BSD500 import BSDS500
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms, datasets



class MMNIST(Dataset):
    def __init__(self, path, bsd500_path, custom_transforms, train=True):
        super(MMNIST, self).__init__()
        self.mnist = datasets.MNIST(Path(path), train=train, transform=custom_transforms, download=True)
        self.bsds = BSDS500(bsd500_path)
        self.rng = np.random.RandomState(42)

        self.transforms = custom_transforms

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = transforms.ToPILImage()(patch)
        patch = self.transforms(patch)
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)

