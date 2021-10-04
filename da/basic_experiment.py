import torch
import itertools
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from train.train import TrainRequest, DADataloader, basic_train
from models.models import resnet18
from da.easy_samples import create_latent_dataloader, \
    get_data_subset, \
    H_collector, \
    get_transport_results, \
    get_trans_feats
import torch.optim as optim


class DADatasets:
    def __init__(self, source_train, source_val, target_train, target_val, n_classes):
        self.source_train = source_train
        self.source_val = source_val
        self.target_train = target_train
        self.target_val = target_val

        self.n_classes = n_classes


def print_stats(stats):
    stats = np.array(stats)
    for i in range(stats.shape[1]):
        print('mean:', np.around(np.mean(stats[:, i]), 3), ', std:', np.around(np.std(stats[:, i]), 3))


def basic_experiment(datasets: DADatasets, epsilon=0.45, m_steps=50, shuffle=True, optimizer=optim.Adam):
    source_train_loader = DataLoader(datasets.source_train, batch_size=128, num_workers=30, shuffle=shuffle)
    source_val_loader = DataLoader(datasets.source_val, batch_size=128, num_workers=30, shuffle=shuffle)

    target_train_loader = DataLoader(datasets.target_train, batch_size=128, shuffle=shuffle)
    target_val_loader = DataLoader(datasets.target_val, batch_size=128, shuffle=shuffle)

    model = resnet18(datasets.n_classes)

    train_request = TrainRequest(model, 10, optimizer=optimizer)
    da_dataloaders = DADataloader(source_train_loader, source_val_loader, target_val_loader)

    basic_train(train_request, da_dataloaders)

    latent_source_train_loader = create_latent_dataloader(model, datasets.source_train)
    latent_source_val_loader = create_latent_dataloader(model, datasets.source_val)
    latent_target_train_loader = create_latent_dataloader(model, datasets.target_train)
    latent_target_val_loader = create_latent_dataloader(model, datasets.target_val)

    stss, ttss, stls, ttls, sts, tts, stl, ttl = get_data_subset(latent_source_train_loader,
                                                                 latent_target_train_loader,
                                                                 latent_source_val_loader.dataset,
                                                                 latent_target_val_loader.dataset,
                                                                 num_samples=100,
                                                                 classes=datasets.n_classes)
    source_train_samples_subset = stss
    target_train_samples_subset = ttss
    source_train_labels_subset = stls
    target_train_labels_subset = ttls
    source_test_samples = sts
    target_test_samples = tts
    source_test_labels = stl
    target_test_labels = ttl

    # Insert sorted dataset into dataloader
    target_subset_dataset = TensorDataset(target_train_samples_subset, target_train_labels_subset)
    target_subset_leader = DataLoader(target_subset_dataset, batch_size=32)

    # Get easy domain and insert into dataloader
    _, easy_samples, _ = H_collector(model.classifier,
                                     target_subset_leader,
                                     isize=[512, 1, 1],
                                     conv=False, m_steps=m_steps,
                                     epsilon=epsilon)

    easy_samples = list(itertools.chain(*easy_samples))
    easy_samples = torch.Tensor(easy_samples)

    # Get labels for semi-supervised Optimal Transport
    target_test_plus_train = torch.cat((target_test_samples.reshape(len(target_test_samples), -1),
                                        target_train_samples_subset.reshape(len(target_train_samples_subset), -1)), 0)

    target_test_plus_train_labels = torch.cat((target_test_labels, target_train_labels_subset.int()), 0)
    semi_target_labels = torch.cat(
        (torch.ones(len(target_test_samples), dtype=torch.float64) * -1, target_train_labels_subset.double()), 0)

    xs_new, xt_new = get_trans_feats(target_test_plus_train, source_test_samples)

    idxs = torch.randint(len(source_test_labels), (100,))
    print('Original transport results')
    get_transport_results(model.classifier, Xs=xs_new, Xt=xt_new[idxs], ys=semi_target_labels,
                          yt=source_test_labels[idxs], ys_test=target_test_plus_train_labels)

    xs_new, xt_new = get_trans_feats(target_test_plus_train, easy_samples)
    print('Easy transport results')
    get_transport_results(model.classifier, Xs=xs_new, Xt=xt_new, ys=semi_target_labels, yt=target_train_labels_subset,
                          ys_test=target_test_plus_train_labels)
