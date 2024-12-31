import torch
import torch.nn as nn
from app.model.pyramid.blocks.pyramid_conv import PyramidConv1D
from app.model.pyramid.blocks.transformer import PositionalEncoding, TransformerEncoder

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x

# Updated PyramidTransformer Model
class PyramidTransformer(nn.Module):
    """
    Pyramid Transformer Model with Single Input.
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_layers: int = 6,       # Increased num_layers from 3 to 6
                 attention_heads: int = 8,  # Increased num_heads from 4 to 8
                 dropout: float = 0.5):
        '''
        :param input_dim: Input feature dimension
        :param num_classes: Number of classes to classify
        :param num_layers: Number of transformer layers
        :param attention_heads: Number of attention heads
        :param dropout: Dropout rate
        '''
        super(PyramidTransformer, self).__init__()
        self.in_channels = 128  # Increased from 64 to 128

        self.dropout = nn.Dropout(dropout)

        # Input convolution with Pyramid Convolution
        self.conv1 = nn.Conv1d(1, 128, kernel_size=7, stride=2, padding=3)  # Increased output channels
        self.pyramid_conv = PyramidConv1D(128, 256, num_layers=num_layers)  # Increased channels
        self.connect_conv = nn.Conv1d(256, 256, kernel_size=1)  # (batch, 256, seq_len/2)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=256)  # Increased embedding dimension

        # Self-Attention
        self.self_attention = SelfAttention(embed_dim=256, num_heads=attention_heads)

        # Transformer Encoder Layers
        self.transformer_encoder = TransformerEncoder(embed_dim=256, num_heads=attention_heads, num_layers=4)  # Increased num_layers from 2 to 4

        # Fully connected layers after pooling
        self.fc = nn.Sequential(
            nn.Linear(256, 512),  # Increased dimensions
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, seq_len)
        x = self.conv1(x.unsqueeze(1))  # (batch, 1, seq_len) -> (batch, 128, seq_len/2)
        x = self.pyramid_conv(x)
        x = self.connect_conv(x)  # (batch, 256, seq_len/2)

        # Prepare for transformer: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len/2, 256)
        x = self.positional_encoding(x)
        x = self.self_attention(x)
        x = self.transformer_encoder(x)

        # Pooling
        x_pooled = torch.mean(x, dim=1)  # (batch, 256)

        # Fully connected layer
        output = self.fc(x_pooled)
        return output
