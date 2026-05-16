"""
EMBEDDING LAYER — Mengubah Token ID jadi Dense Vector

Apa itu embedding?
  - Dalam neural networks, model gak bisa mengerti kata seperti "cat", "dog", dll
  - Kita convert setiap kata jadi number (token ID): cat=5, dog=12, dll
  - Solusi: Embedding — convert token ID jadi dense vector yang meaningful
    cat (ID=5)  → [0.234, -0.512, 0.089, ...]
    dog (ID=12) → [0.201, -0.489, 0.102, ...] 
"""

import numpy as np


class EmbeddingLayer:

    def __init__(self, keras_layer):
        """
        Call keras_layer.get_weights() → return [embedding_matrix]
        Simpan embedding_matrix sebagai self.W
        """
        self.W = keras_layer.get_weights()[0]  # shape: (vocab_size, embed_dim)

    def forward(self, x):
        """
        Forward pass: convert token IDs → embeddings via pure index lookup.
        """
        return self.W[x]  # Pure NumPy index lookup!