import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, batch_first = True):
        super(SelfAttention, self).__init__()
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.attention(x, self.num_heads, self.batch_first)
        x = self.dropout(x)
        return x