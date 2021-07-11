import torch
import torchvision
import os
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
from shallow_network import ShallowNetwork
from network import NeuralNetwork
from attack_network import AttackNetwork

device = "cuda"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


def download_datasets(batch_size_train, batch_size_test):
    train_load = DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=transform),
        batch_size=batch_size_train, shuffle=True)

    test_load = DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=transform),
        batch_size=batch_size_test, shuffle=True)
    return train_load, test_load


def make_train_and_test_loader_for_shallow_model(batch_size_train, batch_size_test):
    train = torchvision.datasets.MNIST("./data", train=True, download=True,
                                       transform=transform)
    test = torchvision.datasets.MNIST("./data", train=False, download=True,
                                      transform=transform)
    _ = 0
    final_data = torch.zeros([70000, 28, 28], dtype=torch.float64)
    final_data[:60000, :, :] = train.train_data
    final_data[60000:, :, :] = test.test_data

    final_labels = torch.zeros(70000, dtype=torch.int64)
    final_labels[:60000] = train.train_labels
    final_labels[60000:] = test.test_labels
    final_data = final_data.numpy()
    final_labels = final_labels.numpy()
    xtrain, xtest, ytrain, ytest = train_test_split(final_data, final_labels, test_size=0.1429)
    train_data_loader = DataLoader(TensorDataset(torch.tensor(xtrain, dtype=torch.float64),
                                                 torch.tensor(ytrain)), batch_size=batch_size_train, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(torch.tensor(xtest, dtype=torch.float64), torch.tensor(ytest)),
                                  batch_size=batch_size_test, shuffle=True)
    return train_data_loader, test_data_loader


def train(dataloader, model, loss_fn, optimizer, epochs, path="./model_weights/mnist_net.pth", train_shallow=False):
    size = len(dataloader.dataset)
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if train_shallow:
                pred = F.log_softmax(pred, 1)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("training finished")

    torch.save(model.state_dict(), path)


def test(dataloader, model, path="./model_weights/mnist_net.pth", shallow=False, print_statement=""):
    model.load_state_dict(torch.load(path))
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            if shallow:
                outputs = F.log_softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(print_statement)
    print("Accuracy : ", 100 * correct / total)


def get_dataset_for_attack_model(train_load, test_load, s_model, train_bs, test_bs,
                                 path="./model_weights/mnist_net_shallow_model.pth"):
    s_model.load_state_dict(torch.load(path))
    train_tensor1 = torch.zeros([len(train_load.dataset), 3], dtype=torch.float64)
    train_tensor2 = torch.zeros([len(test_load.dataset), 3], dtype=torch.float64)
    counter = 0
    with torch.no_grad():
        for data in train_load:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = s_model(images)
            outputs = F.softmax(outputs)
            train_tensor1[counter * train_bs: (counter + 1) * train_bs, :] = torch.topk(outputs, 3).values
            counter += 1

    counter = 0
    with torch.no_grad():
        for data in test_load:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = s_model(images)
            outputs = F.softmax(outputs)
            train_tensor2[counter * test_bs: (counter + 1) * test_bs, :] = torch.topk(outputs, 3).values
            counter += 1

    train_tensor = np.zeros([70000, 3], dtype=np.float64)
    labels = np.zeros(70000, dtype=np.int64)
    labels[len(train_load.dataset):] = 1
    train_tensor[:len(train_load.dataset), :] = train_tensor1.numpy()
    train_tensor[len(train_load.dataset):] = train_tensor2.numpy()
    xtrain, xtest, ytrain, ytest = train_test_split(train_tensor, labels, test_size=0.1429)
    train_data_loader = DataLoader(TensorDataset(torch.tensor(xtrain),
                                                 torch.tensor(ytrain)), batch_size=train_bs, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(torch.tensor(xtest), torch.tensor(ytest)),
                                  batch_size=test_bs, shuffle=True)
    return train_data_loader, test_data_loader


def train_shallow_and_attack_model():

    if not os.path.exists("./membership/model_weights"):
        os.mkdir("./membership/model_weights")
    # train_loader, test_loader = download_datasets(128, 128)
    loss_fn = nn.CrossEntropyLoss()

    # model = NeuralNetwork().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)

    # train(train_loader, model, loss_fn, optimizer, 200,
    #      path="./model_weights/mnist_net.pth")

    shallow_model = ShallowNetwork().to(device)
    shallow_optimizer = torch.optim.Adam(shallow_model.parameters(), lr=1e-3)
    shallow_train_loader, shallow_test_loader = make_train_and_test_loader_for_shallow_model(128, 128)

    train(shallow_train_loader, shallow_model, loss_fn, shallow_optimizer, 200,
          path="./membership/model_weights/mnist_net_shallow_model.pth",
          train_shallow=True)
    test(shallow_test_loader, shallow_model, path="./membership/model_weights/mnist_net_shallow_model.pth",
         shallow=True, print_statement="shallow ")

    shallow_model.load_state_dict(torch.load("./membership/model_weights/mnist_net_shallow_model.pth"))

    attack_train_loader, attack_test_loader = get_dataset_for_attack_model(shallow_train_loader,
                                                                           shallow_test_loader,
                                                                           shallow_model,
                                                                           128, 128)
    attack_model = AttackNetwork().to(device)
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3, weight_decay=0.0001)

    train(attack_train_loader, attack_model, loss_fn, attack_optimizer, 50,
          path="./membership/model_weights/mnist_net_attack_model.pth")
    test(attack_test_loader, attack_model, path="./membership/model_weights/mnist_net_attack_model.pth",
         print_statement="test")


if __name__ == "__main__":
    train_shallow_and_attack_model()
