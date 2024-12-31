from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key):
        attn_output, _ = self.attention(query, key, key)
        output = self.layer_norm(attn_output + query)
        return output