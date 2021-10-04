from da.basic_experiment import basic_experiment, DADatasets
from torchvision import datasets as torch_datasets
from datasets.modern_office31 import ModernOffice31
from data_transforms.data_transforms import small_augment_resnet18_transform
import sys

if __name__ == '__main__':
    data_path = sys.argv[1] + '/Modern-Office-31/'

    domains = ["amazon", "dslr", "synthetic", "webcam"]
    for i in range(len(domains)):
        for j in range(i, len(domains)):
            if i == j:
                continue

            domain1 = domains[i]
            domain2 = domains[j]

            domain1_train = ModernOffice31(data_path, domain1, small_augment_resnet18_transform(3), True)
            domain1_val = ModernOffice31(data_path, domain1, small_augment_resnet18_transform(3), False)

            domain2_train = ModernOffice31(data_path, domain2, small_augment_resnet18_transform(3), True)
            domain2_val = ModernOffice31(data_path, domain2, small_augment_resnet18_transform(3), False)


            print(f'Experiment {domain1} -> {domain2} start')
            print('================================')
            da_datasets = DADatasets(domain1_train, domain1_val, domain2_train, domain2_val, 31)
            basic_experiment(da_datasets)
            print(f'Experiment {domain1} -> {domain2} done')
            print('===============================')

            print(f'Experiment {domain2} -> {domain1} start')
            print('================================')
            da_datasets = DADatasets(domain2_train, domain2_val, domain1_train, domain1_val, 31)
            basic_experiment(da_datasets)
            print(f'Experiment {domain2} -> {domain1} done')
            print('===============================')
