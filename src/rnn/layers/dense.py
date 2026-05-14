import numpy as np
from .activation import get_activation


class DenseLayer:
    """
    Fully-connected layer yang load weights dari Keras Dense layer yang sudah trained.
    """

    def __init__(self, keras_layer):
        """
          1. Get weights dengan keras_layer.get_weights()
          2. Simpan kernel sebagai self.W (weight matrix)
          3. Simpan bias sebagai self.b
          4. Get activation function name dari layer config
        """
        weights = keras_layer.get_weights()
        self.W = weights[0]  # (input_dim, output_dim)
        self.b = weights[1]  # (output_dim,)

        # Get activation dari Keras layer config
        act_name = keras_layer.get_config().get("activation", "linear")
        if isinstance(act_name, dict):
            act_name = act_name.get("class_name", "linear")
        self.activation = get_activation(act_name)

    def forward(self, x):
        """
        Forward pass: linear transformation + activation

        Cara kerjanya:
          1. x @ W        = matrix multiplication
          2. + b          = add bias
          3. activation() = apply nonlinearity
        """
        return self.activation(x @ self.W + self.b)
