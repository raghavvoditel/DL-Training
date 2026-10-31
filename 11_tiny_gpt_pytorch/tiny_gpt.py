# A tiny GPT (decoder-only transformer) that learns to write text one character at a time.
# It trains on story.txt, then generates new text in the same style.
# Run with:  python tiny_gpt.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------- settings ----------------
BLOCK_SIZE = 32     # how many characters the model looks at to predict the next one
EMBED_DIM = 64      # size of each character's vector
NUM_HEADS = 4       # attention heads
NUM_LAYERS = 2      # transformer blocks stacked on top of each other
TRAIN_STEPS = 1500
BATCH_SIZE = 32

# ---------------- data ----------------
text = (Path(__file__).parent / "story.txt").read_text(encoding="utf-8")
chars = sorted(set(text))            # every different character = our vocabulary
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


data = torch.tensor(encode(text))


def get_batch():
    """Pick random pieces of text. Target y = input x shifted one character to the right."""
    starts = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[s:s + BLOCK_SIZE] for s in starts])
    y = torch.stack([data[s + 1:s + BLOCK_SIZE + 1] for s in starts])
    return x, y


# ---------------- model ----------------
class Block(nn.Module):
    """One transformer block: masked self-attention, then a small feed-forward network."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.ff = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
        )

    def forward(self, x):
        T = x.shape[1]
        # Causal mask: True = NOT allowed to look. Blocks every position above the diagonal (the future).
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out               # residual connection ("skip connection")
        x = x + self.ff(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, EMBED_DIM)      # what the character is
        self.position_embed = nn.Embedding(BLOCK_SIZE, EMBED_DIM)   # where it is in the sequence
        self.blocks = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)                # a score for every possible next char

    def forward(self, ids):
        positions = torch.arange(ids.shape[1])
        x = self.token_embed(ids) + self.position_embed(positions)
        x = self.blocks(x)
        return self.head(self.norm(x))  # shape: (batch, time, vocab_size)

    @torch.no_grad()
    def generate(self, start_text, length=200, temperature=0.8):
        ids = torch.tensor([encode(start_text)])
        for _ in range(length):
            context = ids[:, -BLOCK_SIZE:]            # only the last BLOCK_SIZE characters fit
            logits = self(context)[:, -1, :]          # scores for the NEXT character
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)     # sample one character
            ids = torch.cat([ids, next_id], dim=1)
        return decode(ids[0].tolist())


# ---------------- training ----------------
model = TinyGPT()
print(f"Vocabulary: {vocab_size} characters, parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(TRAIN_STEPS + 1):
    x, y = get_batch()
    logits = model(x)
    # cross_entropy wants (N, classes), so flatten batch and time together
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}")

# ---------------- generation ----------------
model.eval()
for temp in (0.5, 1.0):
    print(f"\n--- temperature {temp} (lower = safer, higher = more random) ---")
    print(model.generate("the little robot", length=150, temperature=temp))
