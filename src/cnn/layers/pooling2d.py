from torch import nn
import torch


class MaxPooling2D(nn.Module):
    def __init__(self, keras_layer):
        super().__init__()
        self.pool_size = keras_layer.pool_size
        self.stride = keras_layer.strides

    def forward(self, x):
        '''
        x shape: [B, H, W, C]
        y shape: [B, H_out, W_out, C]
        '''
        B, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        y = torch.zeros((B, H_out, W_out, C), device=x.device)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]  # [B, pH, pW, C]
                y[:, i, j, :] = patch.amax(dim=(1, 2))

        return y


class AveragePooling2D(nn.Module):
    def __init__(self, keras_layer):
        super().__init__()
        self.pool_size = keras_layer.pool_size
        self.stride = keras_layer.strides

    def forward(self, x):
        '''
        x shape: [B, H, W, C]
        y shape: [B, H_out, W_out, C]
        '''
        B, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.stride

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        y = torch.zeros((B, H_out, W_out, C), device=x.device)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]
                y[:, i, j, :] = patch.mean(dim=(1, 2))

        return y
