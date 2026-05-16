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
        kH, kW, C_in, C_out = self.kernel.shape
        B, H, W, C_x = x.shape
        sH, sW = self.stride

        if C_in != C_x:
            raise ValueError(f"Input channels ({C_x}) != kernel channels ({C_in})")

        if self.padding == "same":
            pad_h = max(kH - 1, 0)
            pad_w = max(kW - 1, 0)
            pad_top,    pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left,   pad_right  = pad_w // 2, pad_w - pad_w // 2
            x = np.pad(x, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)), mode='constant')
            H, W = x.shape[1], x.shape[2]

        h_out = (H - kH) // sH + 1
        w_out = (W - kW) // sW + 1

        col = np.lib.stride_tricks.as_strided(
            x,
            shape=(B, h_out, w_out, kH, kW, C_in),
            strides=(
                x.strides[0],
                x.strides[1] * sH,
                x.strides[2] * sW,
                x.strides[1],
                x.strides[2],
                x.strides[3],
            )
        ).reshape(B, h_out * w_out, kH * kW * C_in)

        # kernel shape: [kH*kW*C_in, C_out]
        k = self.kernel.reshape(-1, C_out)

        # matrix multiply: [B, h_out*w_out, C_out]
        y = (col @ k) + self.bias
        y = y.reshape(B, h_out, w_out, C_out)

        return self.activation(y)