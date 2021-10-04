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

    usps_train = torch_datasets.USPS(data_path, train=True, transform=hard_augment_resnet18_transform(1),
                                     download=True)
    usps_val = torch_datasets.USPS(data_path, train=False, transform=hard_augment_resnet18_transform(1),
                                   download=True)

    model = resnet50(10, False)
    print('Experiment MNIST -> USPS start')
    print('================================')
    da_datasets = DADatasets(mnist_train, mnist_val, usps_train, usps_val, 10)
    basic_experiment(model, da_datasets, 0.4, bsize=256)
    print('Experiment MNIST -> USPS done')
    print('===============================')

    model = resnet50(10, False)
    print('Experiment USPS -> MNIST start')
    print('================================')
    da_datasets = DADatasets(usps_train, usps_val, mnist_train, mnist_val, 10)
    basic_experiment(model, da_datasets, 0.4, bsize=256)
    print('Experiment USPS -> MNIST done')
    print('===============================')
