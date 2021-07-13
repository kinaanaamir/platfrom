import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_inversion.face_dataset import FaceDataset
from model_inversion.network import NeuralNetwork

device = "cpu"


def get_data_loaders(batch_size_train, batch_size_test):
    train_set = FaceDataset("./data/faces/training/")
    train_load = DataLoader(
        dataset=train_set, batch_size=batch_size_train,  # shuffle=True,
        num_workers=8, pin_memory=True)
    test_set = FaceDataset("./data/faces/testing/")
    test_load = DataLoader(
        dataset=test_set, batch_size=batch_size_test,  # shuffle=True,
        num_workers=8, pin_memory=True)
    return train_load, test_load


def train(dataloader, model, loss_fn, optimizer, epochs, path="./model_inversion/model_weights/model_inversion.pth",
          train_shallow=False):
    size = len(dataloader.dataset)
    model.train()
    best_loss = 1000000
    for epoch in range(epochs):
        loss_value = 0
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = torch.unsqueeze(X, 1)
            X, y = X.to(device), y.to(device)
            print(X.shape)
            pred = model(X.float())
            print(pred.shape)
            print(pred)
            print(y.shape)
            print(y)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_value += loss

        print(f"loss: ", loss_value / 3.0)
        if best_loss > loss_value / 3.0:
            best_loss = loss_value / 3.0
            torch.save(model.state_dict(), path)
    print("training finished")


def test(dataloader, model, path="./model_inversion/model_weights/mnist_net.pth", shallow=False):
    model.load_state_dict(torch.load(path))
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = torch.unsqueeze(images, 1)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            if shallow:
                outputs = F.log_softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy : ", 100 * correct / total)


def get_target_model_accuracy():
    if not os.path.exists("./model_weights"):
        os.mkdir("./model_weights")
    _, test_loader = get_data_loaders(128, 128)

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load('./model_inversion/model_weights/face_net.pth'))
    test(test_loader, model, path="./model_inversion/model_weights/face_net.pth")


if __name__ == "__main__":
    get_target_model_accuracy()
