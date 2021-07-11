import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, do_log_softmax=True):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.view(-1, 50176)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if not do_log_softmax:
            return x
        return F.log_softmax(x, 1)


if __name__ == "__main__":
    model = NeuralNetwork()

    inp = torch.randn(64, 1, 28, 28)
    torch.save(model.state_dict(), "./model_weights/network.pth")
    model.load_state_dict(torch.load("./model_weights/network.pth"))
    x = model.forward(inp)
    _ = 0
