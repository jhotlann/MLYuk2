import numpy as np
from .activation import get_activation

class LocallyConnected2D:

    def __init__(self, keras_layer):
        '''
        k shape: [output_rows*output_cols, kH*kW*C_in, C_out]
        '''
        weights = keras_layer.get_weights()
        self.kernel = weights[0]
        self.bias = weights[1]
        self.kH, self.kW = keras_layer.kernel_size

        self.activation = get_activation(keras_layer)

    def forward(self, x):
        '''
        x shape: [N, h, w, c]
        '''
        B, H, W, C_in = x.shape
        num_positions, kernel_size, C_out = self.kernel.shape

        H_out = H - self.kH + 1
        W_out = W - self.kW + 1

        y = np.zeros((B, H_out, W_out, C_out), dtype=np.float32)
        for b in range(B):
            pos = 0
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[b, i:i+self.kH, j:j+self.kW, :].reshape(-1)
                    y[b, i, j, :] = patch @ self.kernel[pos] + self.bias[pos]
                    pos += 1

        return self.activation(y)