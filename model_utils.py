import pickle
import numpy as np
import re

# ---------------------------
# Transformer Helper Functions
# ---------------------------
def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        exps = np.exp(x)
        return exps / np.sum(exps)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn = softmax(scores)
    return np.matmul(attn, V), attn

# ---------------------------
# Transformer Classes
# ---------------------------
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, heads * d_k)
        return x

    def forward(self, x, Wq, Wk, Wv, Wo):
        Q = np.matmul(x, Wq)
        K = np.matmul(x, Wk)
        V = np.matmul(x, Wv)
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        out_heads = []
        for h in range(self.num_heads):
            out, _ = scaled_dot_product_attention(Q[:, h, :, :], K[:, h, :, :], V[:, h, :, :])
            out_heads.append(out)
        out = self.combine_heads(np.stack(out_heads, axis=1))
        return np.matmul(out, Wo)

class FeedForward:
    def forward(self, x, W1, b1, W2, b2):
        x = np.matmul(x, W1) + b1
        x = np.maximum(0, x)
        return np.matmul(x, W2) + b2

class EncoderLayer:
    def forward(self, x, mha_weights, ffn_weights, num_heads=4):
        Wq, Wk, Wv, Wo = mha_weights
        W1, b1, W2, b2 = ffn_weights
        mha = MultiHeadAttention(x.shape[-1], num_heads)
        x = layer_norm(x + mha.forward(x, Wq, Wk, Wv, Wo))
        ffn = FeedForward()
        x = layer_norm(x + ffn.forward(x, W1, b1, W2, b2))
        return x

class TransformerModel:
    def __init__(self, params, num_heads=4):
        self.params = params
        self.vocab_size = params["vocab_size"]
        self.seq_len = params["seq_len"]
        self.d_model = params["d_model"]
        self.num_layers = params["num_layers"]
        self.num_heads = num_heads

    def positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return pe

    def forward(self, x):
        x = self.params["embedding_weight"][x]
        if not hasattr(self, 'pos_encoding'):
            self.pos_encoding = self.positional_encoding(self.seq_len, self.d_model)
        x += self.pos_encoding[np.newaxis, :, :]
        for i in range(self.num_layers):
            mha_weights = (
                self.params[f"enc_Wq"][i],
                self.params[f"enc_Wk"][i],
                self.params[f"enc_Wv"][i],
                self.params[f"enc_Wo"][i]
            )
            ffn_weights = (
                self.params[f"enc_ffn_W1"][i],
                self.params[f"enc_ffn_b1"][i],
                self.params[f"enc_ffn_W2"][i],
                self.params[f"enc_ffn_b2"][i]
            )
            encoder = EncoderLayer()
            x = encoder.forward(x, mha_weights, ffn_weights, self.num_heads)
        x = np.mean(x, axis=1)
        return np.matmul(x, self.params["W_out"]) + self.params["b_out"]

# ---------------------------
# Model Loader
# ---------------------------
# ---------------------------
# Model Loader with dynamic seq_len
# ---------------------------
def load_transformer_model(checkpoint_path="transformer_checkpoint.pkl", override_params=None):
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        params = checkpoint["model_params"]
        vocab_local = checkpoint["vocab"]
        label_map_local = checkpoint.get("label_map", {0: "Negative", 1: "Positive"})

        if "<UNK>" not in vocab_local:
            vocab_local["<UNK>"] = max(vocab_local.values()) + 1
        if "<PAD>" not in vocab_local:
            vocab_local["<PAD>"] = max(vocab_local.values()) + 1

        # Apply overrides
        num_heads = 4
        if override_params:
            for k, v in override_params.items():
                if k in params:
                    params[k] = v
                if k == "num_heads":
                    num_heads = v
                if k == "seq_len":
                    params["seq_len"] = v  # update seq_len dynamically

        model_local = TransformerModel(params, num_heads=num_heads)
        return model_local, vocab_local, label_map_local

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, {}, {0: "Negative", 1: "Positive"}


# ---------------------------
# Preprocessing & Prediction
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def encode_and_pad(tokens, vocab, max_len):
    unk_id = vocab.get("<UNK>", 0)
    pad_id = vocab.get("<PAD>", 0)
    ids = [vocab.get(tok, unk_id) for tok in tokens]
    ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

def predict_sentiment(text, model_instance=None, vocab_instance=None, label_map_instance=None):
    model_instance = model_instance or model
    vocab_instance = vocab_instance or vocab
    label_map_instance = label_map_instance or label_map

    if model_instance is None:
        return {"sentiment": "Error", "confidence": 0.0, "probabilities": {"Error": 1.0}}

    tokens = clean_text(text)
    input_ids = encode_and_pad(tokens, vocab_instance, model_instance.seq_len)
    input_ids = np.array([input_ids])
    logits = model_instance.forward(input_ids)
    probs = softmax(logits[0])
    pred_class = int(np.argmax(probs))
    sentiment = label_map_instance.get(pred_class, "Unknown")

    return {
        "sentiment": sentiment,
        "confidence": float(probs[pred_class]),
        "probabilities": {label_map_instance.get(i, f"Class {i}"): float(probs[i]) for i in range(len(probs))}
    }

# ---------------------------
# Load default model
# ---------------------------
model, vocab, label_map = load_transformer_model()
