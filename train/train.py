import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import trange, tqdm
from da.easy_samples import test_model_acc


class DADataloader:
    def __init__(self, source_train_loader, source_val_loader, target_val_loader):
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_val_loader = target_val_loader


class TrainRequest:
    def __init__(self, model, num_epochs, device='cuda',
                 criterion=nn.CrossEntropyLoss,
                 optimizer=optim.Adadelta,
                 scheduler=StepLR):
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.device = device

        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters())
        self.scheduler = scheduler(self.optimizer, step_size=3, gamma=0.7)


def basic_train(train_request: TrainRequest, da_datasets: DADataloader):
    n_samples = len(da_datasets.source_train_loader)
    print(f'Total batches in train dataset: {n_samples}')
    for epoch in trange(train_request.num_epochs):
        train_request.model.train()

        for i, (x, labels) in tqdm(enumerate(da_datasets.source_train_loader)):
            x, labels = x.to(train_request.device), labels.to(train_request.device)
            outputs = train_request.model(x)

            loss = train_request.criterion(outputs, labels)

            train_request.optimizer.zero_grad()
            loss.backward()
            train_request.optimizer.step()

        train_request.scheduler.step()
        train_request.model.eval()

        accuracy = test_model_acc(train_request.model, da_datasets.source_val_loader)
        print(f'Source accuracy: {accuracy}')

        target_accuracy = test_model_acc(train_request.model, da_datasets.target_val_loader)
        print(f'Target accuracy: {target_accuracy}')

        if accuracy > 90:
            print('Stop train, accuracy > 95%')
            return
