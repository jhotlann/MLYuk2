from torch import nn
import torch
from .activation import get_activation

class Dense(nn.Module):
    def __init__(self, keras_layer):
        '''
        weight shape: [input_dim, units]
        bias shape:   [units]
        '''
        super().__init__()
        weights = keras_layer.get_weights()
        self.weight = weights[0]
        self.bias = weights[1]
        self.activation = get_activation(keras_layer)

    def forward(self, x):
        '''
        x shape: [N, input_dim]
        y shape: [N, units]
        '''
        return self.activation(x @ self.weight + self.bias)
