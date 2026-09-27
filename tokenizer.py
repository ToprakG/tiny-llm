import torch

# -------------------------
# Load raw text
# -------------------------

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("characters:", len(text))
print(text[:200])


# -------------------------
# Build vocabulary
# -------------------------

chars = sorted(set(text))

vocab_size = len(chars)

print("vocab size:", vocab_size)
print(chars)


# character <-> integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# -------------------------
# Encode entire dataset
# -------------------------

tokens = torch.tensor(
    [stoi[ch] for ch in text],
    dtype=torch.long
)

print("tokens shape:", tokens.shape)
print("first 50 tokens:", tokens[:50])


# -------------------------
# Decode to verify
# -------------------------

decoded = "".join(itos[i.item()] for i in tokens[:50])

split = int(0.9 * len(tokens))

train_data = tokens[:split]
val_data = tokens[split:]

print("train:", train_data.shape)
print("val:", val_data.shape)

batch_size = 16
block_size = 64

def get_batch(data, batch_size, block_size):
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y

x, y = get_batch(train_data, batch_size, block_size)

print(x.shape)
print(y.shape)