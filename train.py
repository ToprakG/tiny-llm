import torch
import torch.nn.functional as F

from tokenizer import train_data, val_data, get_batch
from model import TinyLLM


# -------------------------
# Configuration
# -------------------------

batch_size = 32
block_size = 128

d_model = 128
n_heads = 4
n_layers = 4

learning_rate = 3e-4
max_steps = 5000


# -------------------------
# Device
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)


# -------------------------
# Model
# -------------------------

model = TinyLLM(
    vocab_size=65,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    max_seq_len=block_size
).to(device)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -------------------------
# Training
# -------------------------

for step in range(max_steps):

    x, targets = get_batch(
        train_data,
        batch_size,
        block_size
    )

    x = x.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print(
            f"step {step}: "
            f"loss {loss.item():.4f}"
        )