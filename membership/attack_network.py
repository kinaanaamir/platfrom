import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackNetwork(nn.Module):
    def __init__(self):
        super(AttackNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(self.fc5(x))


if __name__ == "__main__":
    model = AttackNetwork()

    inp = torch.randn(64, 3)
    x = model.forward(inp)
    _ = 0
