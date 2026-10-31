# Self-attention from the paper "Attention Is All You Need" (2017), written in plain NumPy.
#
#   Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
#
# Q (query)  = what each word is looking for
# K (key)    = what each word offers
# V (value)  = the information each word passes on
# Every word compares its query with every word's key -> scores -> softmax -> weights,
# then takes a weighted average of the values.
# Run with:  python attention.py

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (words x words) similarity table
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # hidden positions get ~0 weight after softmax
    weights = softmax(scores)                  # each row sums to 1
    return weights @ V, weights


# --- A tiny example: a 4-word sentence, each word is a vector of 8 numbers ---
words = ["the", "cat", "sat", "down"]
d_model = 8
X = np.random.randn(len(words), d_model)  # pretend these are word embeddings

# In a real model these 3 matrices are LEARNED during training (here: small random numbers)
W_q = np.random.randn(d_model, d_model) * 0.3
W_k = np.random.randn(d_model, d_model) * 0.3
W_v = np.random.randn(d_model, d_model) * 0.3
Q, K, V = X @ W_q, X @ W_k, X @ W_v

output, weights = attention(Q, K, V)
print("1) Self-attention weights (row = word looking, column = word looked at):")
print("      ", "  ".join(f"{w:>4}" for w in words))
for w, row in zip(words, weights):
    print(f"{w:>5}", row)
print("Output shape:", output.shape, "(one new vector per word)\n")

# --- Causal (masked) attention, used in GPT: a word may only look at itself and earlier words ---
causal_mask = np.tril(np.ones((len(words), len(words)), dtype=bool))
_, masked_weights = attention(Q, K, V, mask=causal_mask)
print("2) Causal attention weights (no looking at future words -> zeros above the diagonal):")
print(masked_weights, "\n")


# --- Multi-head attention: run several smaller attentions in parallel and join them ---
def multi_head_attention(X, num_heads):
    d_head = d_model // num_heads
    heads = []
    for _ in range(num_heads):
        Wq, Wk, Wv = (np.random.randn(d_model, d_head) for _ in range(3))
        out, _ = attention(X @ Wq, X @ Wk, X @ Wv)
        heads.append(out)
    W_o = np.random.randn(d_model, d_model)
    return np.concatenate(heads, axis=-1) @ W_o  # join heads back to d_model numbers


print("3) Multi-head attention (2 heads) output shape:", multi_head_attention(X, num_heads=2).shape)


# --- Positional encoding: attention ignores word order, so we ADD a position signal ---
def positional_encoding(num_positions, d):
    pos = np.arange(num_positions)[:, None]
    i = np.arange(d)[None, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angles), np.cos(angles))  # sin on even, cos on odd


print("\n4) Positional encoding for 4 positions:")
print(positional_encoding(len(words), d_model))
