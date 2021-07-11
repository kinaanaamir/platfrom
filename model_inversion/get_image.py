import torch
from model_inversion.network import NeuralNetwork
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
device = "cuda"


def get_image(label):
    model = NeuralNetwork().to(device)

    model.load_state_dict(torch.load(os.getcwd()+'/model_inversion/model_weights/face_net.pth'))

    image = torch.zeros((1, 1, 112, 92))
    image.requires_grad = True
    y = torch.tensor([label])
    softmax = nn.Softmax()
    alpha = 0.1
    save_image(image, "start_image.png")
    counter = 0
    sample_losses = np.full((1, 25), np.inf)
    best_image = None
    best_loss = np.inf
    while True:
        image = image.to(device)
        output = model(image.float(), False)
        loss = -torch.log(softmax(output)[0][y])
        if best_loss > loss.detach().cpu().clone().numpy():
            best_image = image
            best_loss = loss.detach().cpu().clone().numpy()[0]
        if counter > 25:
            if loss > np.max(sample_losses):
                break
        sample_losses[0, counter % 25] = loss.detach().cpu().clone().numpy()[0]

        derivative = torch.autograd.grad(loss, image)[0]
        image = image - alpha * derivative

        if counter % 1000 == 0:
            print(counter, loss)
        counter += 1
        if counter % 10000 == 0:
            alpha = alpha * 0.99
        if counter > 10000:
            break
    save_image(image, "final_image.png")
    print('finished')


if __name__ == "__main__":
    get_image(0)
