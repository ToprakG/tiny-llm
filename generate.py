import torch
from tokenizer import stoi, itos
from model import TinyLLM


# -------------------------
# Device
# -------------------------

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("device:", device)


# -------------------------
# Load checkpoint
# -------------------------

checkpoint = torch.load(
    "tinyllm.pt",
    map_location=device
)

model = TinyLLM(
    vocab_size=65,
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_seq_len=128
).to(device)


model.load_state_dict(checkpoint)
model.eval()

# -------------------------
# Enter prompt
# -------------------------

prompt = input("Prompt: ")

tokens = torch.tensor(
    [[stoi[c] for c in prompt]],
    dtype=torch.long,
    device=device
)


# -------------------------
# Generate
# -------------------------

max_new_tokens = 100

with torch.inference_mode():
    for _ in range(max_new_tokens):

        # Model predicts the next token
        logits = model(tokens)

        # We only care about the last position
        next_token_logits = logits[:, -1, :]

        # Convert logits -> probabilities
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Sample one token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append it to our sequence
        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )


# -------------------------
# Decode tokens -> text
# -------------------------

generated = "".join(
    itos[token.item()]
    for token in tokens[0]
)

print("\nGenerated:")
print(generated)