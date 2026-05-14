from torch import nn
import torch

class ReLU(nn.Module):
    def forward(self, x):
        return x.clamp(min=0)

class Sigmoid(nn.Module):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class Tanh(nn.Module):
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

class Softmax(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        x_shifted = x - x.amax(dim=self.axis, keepdim=True)
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=self.axis, keepdim=True)

class Linear(nn.Module):
    def forward(self, x):
        return x

class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.3):
        """alpha: negative slope, matches Keras default (0.3)."""
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * x)

ACTIVATION_MAP = {
    "relu":         ReLU,
    "sigmoid":      Sigmoid,
    "tanh":         Tanh,
    "softmax":      Softmax,
    "linear":       Linear,
    "leaky_relu":   LeakyReLU,
}


def get_activation(keras_layer):
    name = keras_layer.activation.__name__
    cls = ACTIVATION_MAP.get(name, Linear)
    return cls()
