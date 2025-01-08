import numpy as np
from torchvision.transforms import Compose
import torch

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return [np.fliplr(x[0]), np.fliplr(x[1])]
        else:
            return x

class RandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            return [np.rot90(x[0],axes=(1,2)), np.rot90(x[1],axes=(1,2))]
        else:
            return x

class ToTensor(object):
    def __call__(self, x):
        return [torch.from_numpy(x[0].copy()), torch.from_numpy(x[1].copy())]
