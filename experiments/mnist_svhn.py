from da.basic_experiment import basic_experiment, DADatasets
from torchvision import datasets as torch_datasets
from data_transforms.data_transforms import hard_augment_resnet18_transform
import sys

if __name__ == '__main__':
    data_path = sys.argv[1]

    mnist_train = torch_datasets.MNIST(data_path, train=True, transform=hard_augment_resnet18_transform(1),
                                       download=True)
    mnist_val = torch_datasets.MNIST(data_path, train=False, transform=hard_augment_resnet18_transform(1),
                                     download=True)

    svhn_train = torch_datasets.SVHN(data_path, split='train', transform=hard_augment_resnet18_transform(3),
                                       download=True)
    svhn_val = torch_datasets.SVHN(data_path, split='test', transform=hard_augment_resnet18_transform(3),
                                     download=True)

    print('Experiment MNIST -> SVHN start')
    print('================================')
    da_datasets = DADatasets(mnist_train, mnist_val, svhn_train, svhn_val, 10)
    basic_experiment(da_datasets, shuffle=True)
    print('Experiment MNIST -> SVHN done')
    print('===============================')

    print('Experiment SVHN -> MNIST start')
    print('================================')
    da_datasets = DADatasets(svhn_train, svhn_val, mnist_train, mnist_val, 10)
    basic_experiment(da_datasets, shuffle=True)
    print('Experiment SVHN -> MNIST done')
    print('===============================')
