import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp_up = nn.Linear(d_model, 4 * d_model)
        self.mlp_down = nn.Linear(4 * d_model, d_model)

    def forward(self, x):

        # -------------------------
        # Attention
        # -------------------------

        residual = x

        x = self.norm1(x)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        B, T, D = Q.shape

        # [B, T, D]
        #      ↓
        # [B, heads, T, d_head]

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # [B, heads, T, T]

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        # causal mask

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        attention = weights @ V

        # [B, heads, T, d_head]
        #       ↓
        # [B, T, D]

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(B, T, D)

        attention = self.out_proj(attention)

        x = residual + attention

        # -------------------------
        # MLP
        # -------------------------

        residual = x

        x = self.norm2(x)

        x = self.mlp_up(x)
        x = F.gelu(x)
        x = self.mlp_down(x)

        x = residual + x

        return x


class TinyLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=128
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits