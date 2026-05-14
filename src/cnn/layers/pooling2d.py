import numpy as np


class MaxPooling2D:
    def __init__(self, keras_layer):
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

        y = np.zeros((B, H_out, W_out, C), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]  # [B, pH, pW, C]
                y[:, i, j, :] = patch.max(axis=(1, 2))

        return y


class AveragePooling2D:
    def __init__(self, keras_layer):
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

        y = np.zeros((B, H_out, W_out, C), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]
                y[:, i, j, :] = patch.mean(axis=(1, 2))

        return y

class GlobalAveragePooling2D:
    def __init__(self, keras_layer=None):
        pass

    def forward(self, x):
        '''
        x shape: [N, H, W, C]
        y shape: [N, C]
        '''
        return x.mean(axis=(1, 2))


class GlobalMaxPooling2D:
    def __init__(self, keras_layer=None):
        pass

    def forward(self, x):
        '''
        x shape: [N, H, W, C]
        y shape: [N, C]
        '''
        return x.max(axis=(1, 2))
