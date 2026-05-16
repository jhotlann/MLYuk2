import numpy as np

# Activation functions: relu, sigmoid, tanh, softmax

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# Lookup table: activation function names → function objects
ACTIVATION_MAP = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "linear": lambda x: x,  # no-op activation
}


def get_activation(name):
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATION_MAP[name]
