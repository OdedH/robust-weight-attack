import torch.nn as nn
import torch
from bitstring import Bits
import torch.nn.functional as F


class Attacked_model(nn.Module):
    def __init__(self, model, arch):
        super(Attacked_model, self).__init__()

        self.model = model
        if isinstance(model, torch.nn.DataParallel):
            self.n_bits = model.module.n_bits
        else:
            self.n_bits = model.n_bits

        if arch[:len("resnet20")] == "resnet20":
            self.w = model.linear.weight.data
            self.b = nn.Parameter(nn.Parameter(model.linear.bias.data), requires_grad=True)
            self.step_size = model.linear.step_size
        elif arch[:len("vgg16_bn")] == "vgg16_bn":
            self.w = model.classifier[6].weight.data
            self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
            self.step_size = model.classifier[6].step_size
        else:
            raise NotImplementedError

        self.w_twos = nn.Parameter(torch.zeros([self.w.shape[0], self.w.shape[1], self.n_bits]), requires_grad=True)

        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        self.reset_w_twos()

    def forward(self, x):

        x = self.model(x)

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] = self.w_twos.data[i][j] + \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])

