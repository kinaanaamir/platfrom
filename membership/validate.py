import torch
import torchvision
import os
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
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
    return correct, total


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
            outputs = s_model(images, False)
            outputs = F.softmax(outputs)
            train_tensor1[counter * train_bs: (counter + 1) * train_bs, :] = torch.topk(outputs, 3).values
            counter += 1

    counter = 0
    with torch.no_grad():
        for data in test_load:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = s_model(images, False)
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

def test_membership_model():
    if not os.path.exists("./membership/model_weights"):
        os.mkdir("./membership/model_weights")
    train_loader, test_loader = download_datasets(128, 128)

    model = NeuralNetwork().to(device)

    attack_train_loader, attack_test_loader = get_dataset_for_attack_model(train_loader, test_loader,
                                                                           model,
                                                                           128, 128,
                                                                           path="./membership/model_weights/mnist_net.pth")
    attack_model = AttackNetwork().to(device)
    attack_model.load_state_dict(torch.load("./membership/model_weights/mnist_net_attack_model.pth"))
    correct, total = test(attack_train_loader, attack_model,
                          path="./membership/model_weights/mnist_net_attack_model.pth",
                          print_statement="test")
    a, b = test(attack_test_loader, attack_model, path="./membership/model_weights/mnist_net_attack_model.pth",
                print_statement="test")
    correct += a
    total += b

    print("Accuracy :", 100 * (correct / total))

if __name__ == "__main__":
    test_membership_model()