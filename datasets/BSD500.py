import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset


class BSDS500(Dataset):
    def __init__(self, path):
        """ 'BSR/BSDS500/data/images'
        """

        image_folder = Path(path)
        self.image_files = list(map(str, image_folder.glob('*/*.jpg')))

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)