import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights

from src.config import LEARNING_RATE, ACCELERATOR


class TorchModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        pass
    
class VGG16Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.vgg16 = vgg16(weights=None, progress=True).features[:]
        self.linear = nn.Linear(512*7*7, 10)
        self.flatten = nn.Flatten()
        

    def forward(self, x):
        x = x.to(ACCELERATOR)
        return self.linear(self.flatten(self.vgg16(x)))