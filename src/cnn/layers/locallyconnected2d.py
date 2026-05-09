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
        self.kernel = weights[0]
        self.bias = weights[1]
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
                    patch = x[b, i:i+self.kH, j:j+self.kW, :].reshape(-1)
                    y[b, i, j, :] = patch @ self.kernel[pos] + self.bias[pos]
                    pos += 1

        return self.activation(y)