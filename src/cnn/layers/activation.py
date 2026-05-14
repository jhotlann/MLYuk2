import numpy as np

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class Tanh:
    def __call__(self, x):
        return np.tanh(x)

class Softmax:
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

class Linear:
    def __call__(self, x):
        return x

class LeakyReLU:
    def __init__(self, alpha=0.3):
        """alpha: negative slope, matches Keras default (0.3)."""
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

ACTIVATION_MAP = {
    "relu":         ReLU,
    "sigmoid":      Sigmoid,
    "tanh":         Tanh,
    "softmax":      Softmax,
    "linear":       Linear,
    "leaky_relu":   LeakyReLU,
}


def get_activation(keras_layer):
    name = keras_layer.activation.__name__
    cls = ACTIVATION_MAP.get(name, Linear)
    return cls()
