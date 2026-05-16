import numpy as np

class Flatten:
    def __init__(self, keras_layer=None):
        pass

    def forward(self, x):
        '''
        x shape: [N, H, W, C]
        y shape: [N, H*W*C]
        '''
        return x.reshape(x.shape[0], -1)
