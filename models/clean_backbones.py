import torch
import torch.nn as nn

from models.utils import PointWiseFeedForward


class CleanSASRecBackbone(nn.Module):
    """Small SASRec encoder with explicit padding and causal masks."""

    def __init__(self, hidden_size, num_layers, num_heads, dropout_rate):
        super().__init__()
        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_layers):
            self.attention_norms.append(nn.LayerNorm(hidden_size, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
            )
            self.ffn_norms.append(nn.LayerNorm(hidden_size, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(hidden_size, dropout_rate))

    def forward(self, seq_emb, item_seq):
        padding_mask = item_seq.eq(0)
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        seq_len = item_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=item_seq.device),
            diagonal=1,
        )

        hidden = seq_emb.transpose(0, 1)
        for attn_norm, attn, ffn_norm, ffn in zip(
            self.attention_norms,
            self.attention_layers,
            self.ffn_norms,
            self.ffn_layers,
        ):
            query = attn_norm(hidden)
            attn_out, _ = attn(
                query,
                hidden,
                hidden,
                attn_mask=causal_mask,
                need_weights=False,
            )
            hidden = query + attn_out
            hidden = hidden.transpose(0, 1)
            hidden = ffn(ffn_norm(hidden))
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            hidden = hidden.transpose(0, 1)

        hidden = hidden.transpose(0, 1)
        return self.last_norm(hidden)
