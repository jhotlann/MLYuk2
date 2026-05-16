import numpy as np
from .activation import sigmoid, tanh


class LSTMLayer:
    """
    LSTM from scratch, loading weights from a trained Keras LSTM layer.
    
    Supports stacking via return_sequences=True and single-step inference
    via forward_step().
    """

    def __init__(self, keras_layer):
        kernel, rec_kernel, bias = keras_layer.get_weights()
        # split fused matrices into 4 gates along the last axis
        (self.W_xi, self.W_xf, self.W_xg, self.W_xo) = np.split(kernel,     4, axis=-1)
        (self.W_hi, self.W_hf, self.W_hg, self.W_ho) = np.split(rec_kernel, 4, axis=-1)
        (self.b_i,  self.b_f,  self.b_g,  self.b_o)  = np.split(bias,       4, axis=-1)
        self.units = self.b_i.shape[0]

    def forward(self, x, h0=None, c0=None, return_sequences=False):
        N, T, _ = x.shape
        h = np.zeros((N, self.units), dtype=x.dtype) if h0 is None else h0.copy()
        c = np.zeros((N, self.units), dtype=x.dtype) if c0 is None else c0.copy()

        if return_sequences:
            out = np.empty((N, T, self.units), dtype=x.dtype)
            for t in range(T):
                h, c = self._cell(x[:, t, :], h, c)
                out[:, t, :] = h
            return out
        else:
            for t in range(T):
                h, c = self._cell(x[:, t, :], h, c)
            return h   # (N, units)

    def forward_step(self, x_t, h, c):
        """
        Returns (new_h, new_c).
        """
        return self._cell(x_t, h, c)

    def _cell(self, x_t, h, c):
        i = sigmoid(x_t @ self.W_xi + h @ self.W_hi + self.b_i)
        f = sigmoid(x_t @ self.W_xf + h @ self.W_hf + self.b_f)
        g = tanh   (x_t @ self.W_xg + h @ self.W_hg + self.b_g)
        o = sigmoid(x_t @ self.W_xo + h @ self.W_ho + self.b_o)
        c_new = f * c + i * g
        h_new = o * tanh(c_new)
        return h_new, c_new
