import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2]).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    model = ShallowNetwork()

    inp = torch.randn(64, 28, 28)
    x = model.forward(inp)
    _ = 0
