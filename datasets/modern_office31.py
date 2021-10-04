import glob
from PIL import Image
from torch.utils.data import Dataset


class ModernOffice31(Dataset):
    """ https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#modern-office-31 """

    def __init__(self, path, domain, transform=None, train=False, train_subset_size=0.7):
        self.labels = {
            'back_pack': 0, 'bike': 1, 'bike_helmet': 2, 'bookcase': 3, 'bottle': 4, 'calculator': 5, 'desk_chair': 6,
            'desk_lamp': 7, 'desktop_computer': 8, 'file_cabinet': 9, 'headphones': 10, 'keyboard': 11,
            'laptop_computer': 12, 'letter_tray': 13, 'mobile_phone': 14, 'monitor': 15, 'mouse': 16, 'mug': 17,
            'paper_notebook': 18, 'pen': 19, 'phone': 20, 'printer': 21, 'projector': 22, 'punchers': 23,
            'ring_binder': 24, 'ruler': 25, 'scissors': 26, 'speaker': 27, 'stapler': 28, 'tape_dispenser': 29,
            'trash_can': 30
        }
        self.supported_domains = {"amazon", "dslr", "synthetic", "webcam"}

        if domain not in self.supported_domains:
            raise RuntimeError(f'Domain "{domain}" not supported, only {self.supported_domains} are supported')

        self.path = path
        self.transform = transform

        self.files = []

        folders = sorted(glob.glob(f'{path}/{domain}/*'))

        for f in folders:
            files = sorted(glob.glob(f'{f}/*'))

            partition = int(len(files) * train_subset_size)

            if train:
                self.files.extend(files[:partition])
            else:
                self.files.extend(files[partition:])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')

        image = self.transform(image)

        return image, self.__label_from_idx(idx)

    def __label_from_idx(self, idx):
        return self.labels[self.files[idx].split('/')[-2]]
