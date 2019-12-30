import math
import numpy
from matplotlib import pyplot as plt

from torch.nn import functional as F

import torch
import torch.nn as nn


class TropixLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = x _trop_ A^T + b`
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, behave_as_tropix=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.behave_as_tropix = behave_as_tropix

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def set_parameters(self, weight: torch.Tensor, bias: torch.Tensor = None):
        self.weight.data = weight.type(torch.FloatTensor)
        if bias is not None:
            self.bias.data = bias.type(torch.FloatTensor)

    def forward(self, input: torch.Tensor):
        if not self.behave_as_tropix:
            return F.linear(input, self.weight, self.bias)

        res = 0

        for a in [-1, 1]:
            input_a = torch.relu(a * input)
            for b in [-1, 1]:
                weight_b = torch.relu(b * self.weight)

                parts = []

                for x in input_a:
                    mm = x * weight_b
                    parts.append(mm.max(dim=1)[0])

                res = res + torch.stack(parts, dim=0) * a * b

        if self.bias is not None:
            return res + self.bias
        else:
            return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def main():
    tpx = TropixLinear(2, 3)  # type: TropixLinear
    tpx.behave_as_tropix = True

    [X1, X2] = numpy.meshgrid(numpy.linspace(-10, 10, 200), numpy.linspace(-10, 10, 200))
    input = torch.tensor(numpy.column_stack((
        X1.flatten(), X2.flatten()
    )), dtype=torch.float32)

    tpx.set_parameters(torch.tensor(numpy.array(
        [[0.1, 0.2], [-0.1, -0.2], [0.1, -0.2]]
    )), bias=torch.tensor([0.0, 0.0, 0.0]))

    r = tpx.forward(input)

    plt.figure()
    plt.contour(X1, X2, numpy.reshape(r.detach()[:, 0].numpy(), X1.shape), levels=40)
    plt.show()


if __name__ == '__main__':
    main()
