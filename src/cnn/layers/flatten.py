from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        x shape: [N, H, W, C]
        y shape: [N, H*W*C]
        '''
        return x.reshape(x.shape[0], -1)
