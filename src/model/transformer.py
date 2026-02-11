"""
This module combines our layernorm, mlp, and attention submodules into a transformer block module.

Input: (B, T, H)
Output: (B, T, H)
"""

from src.model.attention import MHAttention
from src.model.layernorm import LayerNorm
from src.model.mlp import MLP
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1: LayerNorm = LayerNorm(config)
        self.attn: MHAttention = MHAttention(config)
        self.ln2: LayerNorm = LayerNorm(config)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # save residual
        res = x

        # normalize
        h = self.ln1(x)

        # attention
        h = self.attn(h)

        # add residual
        x = res + h

        # repeat for MLP
        res = x
        h = self.ln2(x)
        h = self.mlp(h)
        x = res + h
        return x



class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # get config params
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size

        # transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.num_layers)])

        # token embedding (B, T) -> (B, T, H)
        self.tok_embed = nn.Embedding(self.vocab_size, self.hidden_size, dtype=config.dtype, device=config.device)

        # learned absolute positional embeddings
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_size, dtype=config.dtype, device=config.device)

        # final layer norm
        self.lnf = LayerNorm(config)

        # output head to produce logits (I'm not implementing weight tying for now)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, dtype=config.dtype, device=config.device)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Input length is greater than max_seq_len"
        assert input_ids.dtype == torch.long

        device = input_ids.device

        # token embeddings
        x = self.tok_embed(input_ids)

        # positional embeddings
        pos_ids = torch.arange(T, device=device, dtype=torch.long)    # (T, )
        pos = self.pos_embed(pos_ids).unsqueeze(0)  # (1, T, H)
        x = x + pos                                 # (B, T, H)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # apply last layernorm and output head
        x = self.lnf(x)
        logits = self.lm_head(x)
        return logits
