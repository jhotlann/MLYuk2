from torch import nn
import torch

def corr2d(X, K):
    """
    Compute 2D cross-correlation.
    X: input
    K: kernel 
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device = X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel, bias):
        super().__init__()
        self.kernel = kernel
        self.bias = bias
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        x shape: [N, h, w, c]
        y shape: 
        '''
        kH, kW, C_in, C_out = self.kernel.shape
        B, H, W, C_x = x.shape

        if C_in != C_x:
            raise ValueError(
                f"Input channels ({C_x}) "
                f"!= kernel channels ({C_in})"
            )

        h_out = H - kH + 1
        w_out = W - kW + 1
  
        y = torch.zeros((B, h_out, w_out, C_out), device = x.device)
        for b in range(B):
            for out_c in range(C_out) :
                y_in = torch.zeros((h_out, w_out), device = x.device)
                for in_c in range(C_in):
                    x_in = x[b, :, :, in_c]
                    k_in = self.kernel[:,:, in_c,out_c]
                    y_in += corr2d(x_in,k_in)
                y_in += self.bias[out_c]
                y[b, :, :, out_c] = self.activation(y_in)

        return y

