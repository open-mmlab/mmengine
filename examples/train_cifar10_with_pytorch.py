import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward_train(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_test(self, x):
        with torch.no_grad():
            x = self.forward_train(x)
        return x

    def forward(self, data_batch, return_loss=False):
        inputs, labels = zip(*data_batch)
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        if return_loss:
            outputs = self.forward_train(inputs)
            loss = self.criterion(outputs, labels)
            return {'loss': loss, 'log_vars': {'loss': loss.item()}}
        else:
            outputs = self.forward_test(inputs)
            predictions = torch.argmax(outputs, 1)
            return predictions


def train(model, train_loader, optimizer, max_epochs, val_loader=None):
    running_loss = 0.0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_idx, data_batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(data_batch, return_loss=True)
            outputs['loss'].backward()
            optimizer.step()

            running_loss += outputs['log_vars']['loss']
            if batch_idx % 50 == 0:
                print(f'trian: [{epoch}/{max_epochs}] epochs, '
                      f'[{batch_idx}/{len(train_loader)}] iterations, '
                      f'lr: {optimizer.param_groups[0]["lr"]}, '
                      f'loss: {running_loss / 50:.5f}')
                running_loss = 0.0

        if val_loader is not None:
            if epoch % 2 == 0:
                test(model, val_loader)


def test(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data_batch in test_loader:
            labels = [data[1] for data in data_batch]
            preditions = model(data_batch)
            labels_np = np.array(labels)
            preditions_np = preditions.cpu().numpy()
            total += len(labels)
            correct += (labels_np == preditions_np).sum()

    print(f'test: accuracy: {100 * correct // total}%')


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x)

    model = ToyModel()
    if torch.cuda.is_available():
        model = model.to('cuda')

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, train_loader, optimizer, 5, val_loader=test_loader)
    test(model, test_loader)


if __name__ == '__main__':
    main()
