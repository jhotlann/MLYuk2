import numpy as np
from .activation import get_activation

class Conv2D:

    def __init__(self, keras_layer):
        '''
        k shape: [kH, kW, C_in, C_out]
        '''
        weights = keras_layer.get_weights()
        self.kernel = weights[0]
        self.bias = weights[1]

        self.activation = get_activation(keras_layer)

        self.stride = keras_layer.strides
        self.padding = keras_layer.padding

        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape

    def forward(self, x):
        '''
        x shape: [B, H, W, C_in]
        y shape: [B, H_out, W_out, C_out]
        '''
        kH, kW, C_in, C_out = self.kernel.shape
        B, H, W, C_x = x.shape
        sH, sW = self.stride

        if C_in != C_x:
            raise ValueError(
                f"Input channels ({C_x}) "
                f"!= kernel channels ({C_in})"
            )

        if self.padding == "same":
            pad_h = max(kH - 1, 0)
            pad_w = max(kW - 1, 0)
            pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

            # NumPy padding
            x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            H, W = x.shape[1], x.shape[2]

        h_out = (H - kH) // sH + 1
        w_out = (W - kW) // sW + 1

        y = np.zeros((B, h_out, w_out, C_out), dtype=np.float32)
        for b in range(B):
            for out_c in range(C_out):
                y_in = np.zeros((h_out, w_out), dtype=np.float32)
                for in_c in range(C_in):
                    x_in = x[b, :, :, in_c]
                    k_in = self.kernel[:, :, in_c, out_c]

                    for i in range(h_out):
                        for j in range(w_out):
                            y_in[i, j] += (
                                x_in[i*sH:i*sH+kH, j*sW:j*sW+kW] * k_in
                            ).sum()
                y_in += self.bias[out_c]
                y[b, :, :, out_c] = self.activation(y_in)

        return y