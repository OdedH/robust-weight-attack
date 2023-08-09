import torch
from torch import nn
import kornia.augmentation as K


class RWAugmentations(nn.Module):
    def __init__(self, transforms: list, p: float = 1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
        for transform in self.transforms:
            x = transform(x)
        return x

