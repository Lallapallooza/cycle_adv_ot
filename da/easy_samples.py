import numpy as np
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import itertools
import ot

def test_model_acc(model, dataloader: DataLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            x, y = data
            outputs = model(x.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y.cuda()).sum().item()
    return 100 * correct / total


def create_latent_dataloader(model, data, batch_size=256, num_workers=30):
    feats = []
    labels = []
    dataloader_orig = DataLoader(data, batch_size=batch_size, num_workers=num_workers)
    for sample in tqdm(dataloader_orig):
        labels.append(sample[1])
        feats.append(model.forward_seq(sample[0].cuda()).cpu().detach())

    dataset = TensorDataset(torch.cat(feats), torch.cat(labels))
    return DataLoader(dataset, batch_size=1, num_workers=num_workers)


def get_trans_feats(*raw_feats):
    return [f + 1e-7 for f in raw_feats]


def get_transport_results(model, Xs, Xt, ys, yt, ys_test):
    print('Xs', len(Xs))
    print('Xt', len(Xt))
    print('ys', len(ys))
    print('yt', len(Xt))

    transports = {'EMD': ot.da.EMDTransport(),
                  'Sinkhorn': ot.da.SinkhornTransport(reg_e=4, verbose=False),
                  'SinkhornLpl1': ot.da.SinkhornLpl1Transport(reg_e=4, reg_cl=0.1, verbose=False),
                  'SinkhornL1l2': ot.da.SinkhornL1l2Transport(reg_e=4, reg_cl=0.1, verbose=False),
                  #'MapOT': ot.da.MappingTransport(kernel="linear", mu=1, eta=1e-0, bias=True, max_iter=20,
                  #                                verbose=False)
                 }

    accs = []
    for ot_name in transports:
        ot_mapping = transports[ot_name]
        try:
            ot_mapping.fit(Xs=Xs[:].numpy(),
                           Xt=Xt[:].numpy(),
                           ys=ys,
                           yt=yt)
            # for out of source samples, transform applies the linear mapping
            X_test_mapped = ot_mapping.transform(Xs=Xs.numpy())
            X_test_mapped = TensorDataset(torch.FloatTensor(X_test_mapped), ys_test)
            X_test_mapped = DataLoader(X_test_mapped, batch_size=100)
            print(ot_name)
            accs.append(test_model_acc(model, X_test_mapped))
            print(accs[-1])
            print('--------------------------')
        except:
            print(f'Failed for {ot_name}, try change reg_e parameter')

    return accs



def H_collector(model, dataloader, isize, m_steps=10, epsilon=0.01, conv=True):
    device = 'cuda'
    num_batches = dataloader.__len__()
    model.eval()
    real_samples = []
    easy_samples = []
    y_s = []
    correct = 0
    total = 0

    for data in tqdm(dataloader):
        x, y = data
        if conv:
            ori_img = x.reshape(x.size(0), isize[0], isize[1], isize[2])
        else:
            ori_img = x.reshape(x.size(0), isize[0] * isize[1] * isize[2])
        ori_img = ori_img.to(device)
        img = ori_img.clone()

        for i in range(m_steps):
            img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
            img_x.requires_grad_(True)

            output = model(img_x)
            loss = nn.CrossEntropyLoss()(output, y.to(device))

            model.zero_grad()
            loss.backward()

            input_grad = img_x.grad.data
            input_grad = -input_grad

            img = img.data + epsilon * torch.sign(input_grad)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        outputs = model(img)

        real_samples.append(x.cpu().data.numpy())
        easy_samples.append(img.cpu().data.numpy())

        y_s.append(y.cpu().data.numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y.cuda()).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))
    return real_samples, easy_samples, y_s


def get_easy_dataset(model, H_divergence, train_loader, epsilon):
    real_samples, easy_samples, y_s = H_collector(model, H_divergence, train_loader, epsilon)

    flatten_easy_samples = list(itertools.chain(*easy_samples))
    y_s = list(itertools.chain(*y_s))
    drop_indexes = drop_nan(flatten_easy_samples)
    if len(drop_indexes) > 0:
        print(len(drop_indexes))
        flatten_easy_samples = [v for i, v in enumerate(flatten_easy_samples) if i not in drop_indexes]
        y_s = [v for i, v in enumerate(y_s) if i not in drop_indexes]

    flatten_easy_samples = torch.Tensor(flatten_easy_samples)
    y_s = torch.LongTensor(y_s)

    easy_train_dataset = TensorDataset(flatten_easy_samples, y_s)
    easy_train_loader = DataLoader(easy_train_dataset, batch_size=150)

    return easy_train_dataset, easy_train_loader


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def drop_nan(list_):
    index = 0
    drop_indexes = []
    for x in list_:
        bool_matrix = np.isnan(x)
        if 1 in bool_matrix:
            drop_indexes.append(index)
        index += 1
    return drop_indexes


def get_subset(data_loader, per_class=10, classes=10):
    subset_dataset = []
    class_counter = Counter()
    for batch_i, (x, y) in enumerate(data_loader):
        # print(y)
        if y < classes:
            if class_counter[y.item()] < per_class:
                class_counter[y.item()] += 1
                subset_dataset.append((x, y))
            if all([x == per_class for x in class_counter.values()]):
                break
    return subset_dataset


def takeSecond(elem):
    return elem[1]


def get_data_subset(source_train_loader, target_train_loader, source_val, target_val, num_samples=100, classes=10):
    source_subset = get_subset(source_train_loader, per_class=num_samples / classes, classes=classes)
    target_subset = get_subset(target_train_loader, per_class=num_samples / classes, classes=classes)

    source_subset.sort(key=takeSecond)
    target_subset.sort(key=takeSecond)

    # Get structured samples
    source_list_subset = [source_subset[n][0].numpy() for n in range(num_samples)]
    target_list_subset = [target_subset[n][0].numpy() for n in range(num_samples)]

    source_list_subset = torch.Tensor(source_list_subset)
    target_list_subset = torch.Tensor(target_list_subset)

    source_labels_subset = [source_subset[n][1] for n in range(num_samples)]
    target_labels_subset = [target_subset[n][1] for n in range(num_samples)]

    source_labels_subset = torch.from_numpy(np.array((source_labels_subset)))
    target_labels_subset = torch.from_numpy(np.array((target_labels_subset)))

    # Get test samples
    source_test_list = [source_val[n][0].numpy() for n in range(len(source_val))]
    target_test_list = [target_val[n][0].numpy() for n in range(len(target_val))]

    source_test_list = torch.Tensor(source_test_list)
    target_test_list = torch.Tensor(target_test_list)

    source_test_labels = [source_val[n][1] for n in range(len(source_val))]
    target_test_labels = [target_val[n][1] for n in range(len(target_val))]

    source_test_labels = torch.from_numpy(np.array((source_test_labels)))
    target_test_labels = torch.from_numpy(np.array((target_test_labels)))

    return source_list_subset, \
           target_list_subset, \
           source_labels_subset, \
           target_labels_subset, \
           source_test_list, \
           target_test_list, \
           source_test_labels, \
           target_test_labels
