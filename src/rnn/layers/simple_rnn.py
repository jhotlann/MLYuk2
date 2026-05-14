import numpy as np


class SimpleRNNLayer:
    """
    SimpleRNN cell implementation
    Load weights dari Keras SimpleRNN layer yang sudah trained.
    """

    def __init__(self, keras_layer):
        W_x, W_h, b = keras_layer.get_weights()
        self.W_x = W_x   # shape: (input_dim, units)
        self.W_h = W_h   # shape: (units, units)
        self.b   = b     # shape: (units,)
        self.units = b.shape[0]

    def forward(self, x, h0=None, return_sequences=False):
        N, T, _ = x.shape
        h = np.zeros((N, self.units), dtype=x.dtype) if h0 is None else h0.copy()

        if return_sequences:
            # Simpan semua timesteps
            out = np.empty((N, T, self.units), dtype=x.dtype)
            for t in range(T):
                h = np.tanh(x[:, t, :] @ self.W_x + h @ self.W_h + self.b)
                out[:, t, :] = h
            return out
        else:
            # Hanya return last timestep
            for t in range(T):
                h = np.tanh(x[:, t, :] @ self.W_x + h @ self.W_h + self.b)
            return h   # shape: (N, units)

    def forward_step(self, x_t, h):
        return np.tanh(x_t @ self.W_x + h @ self.W_h + self.b)
