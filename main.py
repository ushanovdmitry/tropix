import numpy as np
import math

import matplotlib
matplotlib.use("qt5agg")

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from torch.optim.rmsprop import RMSprop
# from torch.optim.optimizer import Optimizer
from torch.optim import Adam


class TrueFunc(nn.Module):
    dim_in = 1
    dim_out = 1

    def forward(self, y) -> torch.Tensor:
        # return y ** 3
        return 3 * y ** 2 - torch.sin(y * 5) * (y - 0.2)


class Func(nn.Module):
    def __init__(self):
        super(Func, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(TrueFunc.dim_in, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, TrueFunc.dim_out)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)


def main():
    func = Func()

    # optimizer = RMSprop(func.parameters(), lr=1e-2)
    optimizer = Adam(func.parameters())

    losses = []

    fig = plt.figure(figsize=(12, 4), facecolor='white')  # type: plt.Figure
    ax = fig.add_subplot(111)

    for i in range(2500):
        y = torch.linspace(-1.0, 1.0, 15).view(-1, 1)
        y = y + torch.randn_like(y) * (1 - math.tanh(10.0 * i / 2500.0))
        z = TrueFunc().forward(y)

        z_ = func(y)
        loss = torch.nn.functional.mse_loss(z, z_)  # type: torch.Tensor
        losses.append(loss.detach().item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            with torch.no_grad():
                y = torch.linspace(-1.0, 1.0, 100).view(-1, 1)
                z = TrueFunc().forward(y)
                z_ = func(y)

            ax.clear()
            plt.title(f"iter {i}")
            ax.plot(y.detach().numpy(), z.detach().numpy(), '-')
            ax.plot(y.detach().numpy(), z_.detach().numpy(), '-')
            plt.draw()
            plt.pause(0.05)

    plt.show()
    # plt.close(fig)
    # return np.log(losses)


if __name__ == '__main__':
    main()
