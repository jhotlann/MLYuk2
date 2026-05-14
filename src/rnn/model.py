import numpy as np
from .layers.dense import DenseLayer
from .layers.embedding import EmbeddingLayer
from .layers.simple_rnn import SimpleRNNLayer
from .layers.lstm import LSTMLayer


class CaptioningModelScratch:
    """
    Full pre-inject image captioning pipeline from scratch (NumPy only).

    Parameters
    keras_model : compiled Keras model (output of build_decoder_keras())
    word2idx    : dict[str, int]
    idx2word    : dict[int, str]
    decoder_type: "rnn" or "lstm"
    num_rnn_layers: number of stacked recurrent layers in the Keras model
    """

    def __init__(self, keras_model, word2idx, idx2word, decoder_type="lstm", num_rnn_layers=1):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.decoder_type = decoder_type.lower()
        self.num_rnn_layers = num_rnn_layers

        self._load_layers(keras_model)


    def _load_layers(self, keras_model):
        layer_map = {l.name: l for l in keras_model.layers}

        # Dense projection: CNN feature → embed_dim
        proj_key = self._find_layer(layer_map, "dense_proj", "dense", exclude_suffix="out")
        self.dense_proj = DenseLayer(layer_map[proj_key])

        # Embedding
        embedding_key = self._find_layer(layer_map, "embedding")
        self.embedding = EmbeddingLayer(layer_map[embedding_key])

        # Recurrent layers (stacked)
        self.rnn_layers = []
        rnn_class = SimpleRNNLayer if self.decoder_type == "rnn" else LSTMLayer
        rnn_type_name = "simple_rnn" if self.decoder_type == "rnn" else "lstm"

        for i in range(self.num_rnn_layers):
            suffix = "" if i == 0 else f"_{i}"
            key = self._find_layer(layer_map, f"{rnn_type_name}{suffix}", rnn_type_name)
            self.rnn_layers.append(rnn_class(layer_map[key]))

        # Dense output: hidden → vocab_size (softmax)
        out_key = self._find_layer(layer_map, "dense_out", "dense_1", "dense")
        self.dense_out = DenseLayer(layer_map[out_key])

    def _find_layer(self, layer_map, *candidates):
        for c in candidates:
            if c in layer_map:
                return c
        # fallback: search by partial match of first candidate
        for name in layer_map:
            if candidates[0].split("_")[0] in name:
                return name
        raise KeyError(f"Could not find layer matching any of {candidates} in {list(layer_map)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_caption(self, image_feature, max_len=25):
        """
        Greedy decode a single image.

        image_feature : (2048,) or (1, 2048) numpy array from InceptionV3
        Returns       : str — generated caption (without <start>/<end>/<pad>)
        """
        feat = np.array(image_feature, dtype=np.float32).reshape(1, -1)  # (1, 2048)

        # x_{-1}: project CNN feature to embed_dim
        x_neg1 = self.dense_proj.forward(feat)   # (1, embed_dim)

        # Initialise hidden (and cell) state
        if self.decoder_type == "lstm":
            hs = [np.zeros((1, l.units), dtype=np.float32) for l in self.rnn_layers]
            cs = [np.zeros((1, l.units), dtype=np.float32) for l in self.rnn_layers]
        else:
            hs = [np.zeros((1, l.units), dtype=np.float32) for l in self.rnn_layers]

        # Feed x_{-1} through all recurrent layers (t = -1)
        inp = x_neg1   # (1, embed_dim)
        if self.decoder_type == "lstm":
            for i, layer in enumerate(self.rnn_layers):
                hs[i], cs[i] = layer.forward_step(inp, hs[i], cs[i])
                inp = hs[i]
        else:
            for i, layer in enumerate(self.rnn_layers):
                hs[i] = layer.forward_step(inp, hs[i])
                inp = hs[i]

        # Greedy decode
        token = self.word2idx.get("<start>", 1)
        tokens = []

        for _ in range(max_len):
            x_t = self.embedding.forward(np.array([[token]]))[:, 0, :]  # (1, embed_dim)

            inp = x_t
            if self.decoder_type == "lstm":
                for i, layer in enumerate(self.rnn_layers):
                    hs[i], cs[i] = layer.forward_step(inp, hs[i], cs[i])
                    inp = hs[i]
            else:
                for i, layer in enumerate(self.rnn_layers):
                    hs[i] = layer.forward_step(inp, hs[i])
                    inp = hs[i]

            logits = self.dense_out.forward(inp)   # (1, vocab_size)
            token  = int(np.argmax(logits[0]))

            word = self.idx2word.get(token, "<unk>")
            if word in ("<end>", "<pad>"):
                break
            tokens.append(word)

        return " ".join(tokens)

    def generate_captions_batch(self, image_features, max_len=25):
        """
        Greedy decode a batch of images. Calls generate_caption per image.
        image_features : (N, 2048)
        Returns        : list[str] of length N
        """
        return [self.generate_caption(image_features[i], max_len=max_len)
                for i in range(len(image_features))]
