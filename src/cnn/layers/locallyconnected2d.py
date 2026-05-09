class LocallyConnected2D(nn.Module):
    ACTIVATION_MAP = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "linear": nn.Identity(),
        "softmax": nn.Softmax(dim=-1),
    }

    def __init__(self, keras_layer):
        '''
        k shape: [output_rows*output_cols, kH*kW*C_in, C_out]
        '''
        super().__init__()
        weights = keras_layer.get_weights()
        # Keras shape: [output_rows, output_cols, input_size, C_out]
        # reshape to: [output_rows*output_cols, input_size, C_out]
        w = weights[0]
        rows, cols, input_size, C_out = w.shape
        self.kernel = w.reshape(rows * cols, input_size, C_out)
        self.bias = weights[1].reshape(rows * cols, C_out)
        self.kH, self.kW = keras_layer.kernel_size

        act_name = keras_layer.activation.__name__
        self.activation = self.ACTIVATION_MAP.get(act_name, nn.Identity())

    def forward(self, x):
        '''
        x shape: [N, h, w, c]
        '''
        B, H, W, C_in = x.shape
        num_positions, kernel_size, C_out = self.kernel.shape

        H_out = H - self.kH + 1
        W_out = W - self.kW + 1

        y = torch.zeros((B, H_out, W_out, C_out), device=x.device)
        for b in range(B):
            pos = 0
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[b, i:i+self.kH, j:j+self.kW, :]
                    patch = patch.reshape(-1)

                    for out_c in range(C_out):
                        k = self.kernel[pos, :, out_c]
                        y[b, i, j, out_c] = (
                            patch * k
                        ).sum() + self.bias[pos, out_c]
                    pos += 1

        return self.activation(y)