from torch import nn
import torch

class Conv2D(nn.Module):
    ACTIVATION_MAP = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "linear": nn.Identity(),
        "softmax": nn.Softmax(dim=-1),
    }

    def __init__(self, keras_layer):
        '''
        k shape: [kH, kW, C_in, C_out]
        '''
        super().__init__()
        weights = keras_layer.get_weights()
        self.kernel = weights[0]
        self.bias = weights[1]

        act_name = keras_layer.activation.__name__
        self.activation = self.ACTIVATION_MAP.get(act_name, nn.Identity())

        self.stride = keras_layer.strides
        self.padding = keras_layer.padding

        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape

    def forward(self, x):
        '''
        x shape: [N, H, W, C_in]
        y shape: [N, H_out, W_out, C_out]
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

            x = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (pad_left, pad_right, pad_top, pad_bottom)).permute(0, 2, 3, 1)
            H, W = x.shape[1], x.shape[2]

        h_out = (H - kH) // sH + 1
        w_out = (W - kW) // sW + 1

        y = torch.zeros((B, h_out, w_out, C_out), device=x.device)
        for b in range(B):
            for out_c in range(C_out):
                y_in = torch.zeros((h_out, w_out), device=x.device)
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