from datasets.MMNIST import MMNIST
from da.basic_experiment import basic_experiment, DADatasets
from torchvision import datasets as torch_datasets
from data_transforms.data_transforms import hard_augment_resnet18_transform
import sys
from models.models import resnet50

if __name__ == '__main__':
    data_path = sys.argv[1]
    bsd_path = data_path + '/BSR/BSDS500/data/images'

    mnist_train = torch_datasets.MNIST(data_path, train=True, transform=hard_augment_resnet18_transform(1),
                                       download=True)
    mnist_val = torch_datasets.MNIST(data_path, train=False, transform=hard_augment_resnet18_transform(1),
                                     download=True)

    mmnist_train = MMNIST(data_path, bsd_path, train=True, custom_transforms_mnist=hard_augment_resnet18_transform(1), custom_transforms_bsd=hard_augment_resnet18_transform(3))
    mmnist_val = MMNIST(data_path, bsd_path, train=False, custom_transforms_mnist=hard_augment_resnet18_transform(1), custom_transforms_bsd=hard_augment_resnet18_transform(3))

    model = resnet50(10, False)
    print('Experiment MNIST -> MMNIST start')
    print('================================')
    da_datasets = DADatasets(mnist_train, mnist_val, mmnist_train, mmnist_val, 10)
    basic_experiment(model, da_datasets, bsize=256)
    print('Experiment MNIST -> MMNIST done')
    print('===============================')

    model = resnet50(10, False)
    print('Experiment MMNIST -> start')
    print('================================')
    da_datasets = DADatasets(mmnist_train, mmnist_val, mnist_train, mnist_val, 10)
    basic_experiment(model, da_datasets, bsize=256)
    print('Experiment MMNIST -> MNIST done')
    print('===============================')
