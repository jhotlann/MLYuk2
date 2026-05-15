import numpy as np


def _pool_windows(x, pool_size, stride):
    B, H, W, C = x.shape
    pH, pW = pool_size
    sH, sW = stride

    H_out = (H - pH) // sH + 1
    W_out = (W - pW) // sW + 1

    if sH == pH and sW == pW:
        x_crop = x[:, :H_out*sH, :W_out*sW, :]
        return x_crop.reshape(B, H_out, pH, W_out, pW, C)

    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(B, H_out, W_out, pH, pW, C),
        strides=(
            x.strides[0],
            x.strides[1] * sH,
            x.strides[2] * sW,
            x.strides[1],
            x.strides[2],
            x.strides[3],
        )
    )
    return np.ascontiguousarray(windows)


class MaxPooling2D:
    def __init__(self, keras_layer):
        self.pool_size = keras_layer.pool_size
        self.stride    = keras_layer.strides

    def forward(self, x):
        windows = _pool_windows(x, self.pool_size, self.stride)
        ax = (2, 4) if self.stride == self.pool_size else (3, 4)
        return windows.max(axis=ax)


class AveragePooling2D:
    def __init__(self, keras_layer):
        self.pool_size = keras_layer.pool_size
        self.stride    = keras_layer.strides

    def forward(self, x):
        windows = _pool_windows(x, self.pool_size, self.stride)
        ax = (2, 4) if self.stride == self.pool_size else (3, 4)
        return windows.mean(axis=ax)

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
