import numpy as np
from .activation import get_activation

class LocallyConnected2D:
    def __init__(self, keras_layer):
        self.kernel      = keras_layer._lc_kernel
        self.bias        = keras_layer._lc_bias
        self.kH, self.kW = keras_layer.kernel_size
        self.C_out       = self.kernel.shape[-1]
        self.activation  = get_activation(keras_layer)
        self.stride      = keras_layer.strides
        self.padding     = keras_layer.padding

    def forward(self, x):
        B, H, W, C_in = x.shape
        kH, kW = self.kH, self.kW
        sH, sW = self.stride

        if self.padding == 'same':
            pad_h = max(kH - 1, 0)
            pad_w = max(kW - 1, 0)
            x = np.pad(x,
                       ((0,0),
                        (pad_h//2, pad_h - pad_h//2),
                        (pad_w//2, pad_w - pad_w//2),
                        (0,0)),
                       mode='constant')
            H, W = x.shape[1], x.shape[2]

        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1
        n_pos = H_out * W_out

        if self.kernel.ndim == 2:
            self.kernel = np.tile(self.kernel[np.newaxis], (n_pos, 1, 1))  # (n_pos, kH*kW*C_in, C_out)
            self.bias   = np.tile(self.bias[np.newaxis],   (n_pos, 1))     # (n_pos, C_out)

        y = np.zeros((B, H_out, W_out, self.C_out), dtype=np.float32)

        pos = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+kH, j*sW:j*sW+kW, :].reshape(B, -1)
                y[:, i, j, :] = patch @ self.kernel[pos] + self.bias[pos]
                pos += 1

        return self.activation(y)