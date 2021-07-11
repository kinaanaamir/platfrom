import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool_1 = nn.MaxPool2d(2)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(41216, 128)
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x, do_log_softmax=True):
        x = F.relu(self.conv_1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.maxpool_2(x)
        x = x.view(-1, 41216)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if not do_log_softmax:
            return x
        return F.log_softmax(x, 1)


if __name__ == "__main__":
    model = NeuralNetwork()

    inp = torch.randn(64, 1, 112, 92)
    x = model.forward(inp)
    _ = 0
