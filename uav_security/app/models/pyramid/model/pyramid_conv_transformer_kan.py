import torch
import torch.nn as nn

from app.models.pyramid.blocks.dilated_conv import DilatedConv1D
from app.models.pyramid.blocks.kan import KANLinear
from app.models.pyramid.blocks.pyramid_conv import PyramidConv1D
from app.models.pyramid.blocks.transformer import PositionalEncoding, TransformerEncoder
from app.models.pyramid.model.pyramid_conv_lstm import CrossAttention

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


class PyramidTransformerKAN(nn.Module):
    """
    Pyramid Transformer Model for Cyber-Physical Systems.
    """
    def __init__(self,
                 input_dim_cyber,
                 input_dim_physical,
                 num_classes,
                 num_layers:int  = 3,
                 attention_heads : int = 4,
                 dropout: float = 0.5):
        '''
        :param input_dim_cyber:  cyber data input feature dimension
        :param input_dim_physical: physical data input feature dimension
        :param num_classes:  Number of classes to classify
        :param num_layers: transformer num_layers
        :param attention_heads: Number of attention heads
        :param dropout: Dropout rate
        '''
        super(PyramidTransformerKAN, self).__init__()
        self.in_channels_cyber = 64
        self.in_channels_physical = 64

        self.dropout = nn.Dropout(dropout)

        # Cyber branch with Pyramid Convolution
        self.conv1_cyber = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pyramid_cyber = PyramidConv1D(64, 128, num_layers=num_layers)
        self.connect_cyber = nn.Conv1d(128, 128, kernel_size=1) # (batch, 128, seq_len/2)

        # Physical branch with Dilated Convolution
        self.conv1_physical = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.dilated_physical = DilatedConv1D(64, 128, dilation=2, num_layers=num_layers)
        self.connect_physical = nn.Conv1d(128, 128, kernel_size=1)

        # Positional Encoding
        self.positional_encoding_cyber = PositionalEncoding(d_model=128)
        self.positional_encoding_physical = PositionalEncoding(d_model=128)

        # Self-Attention within each branch
        self.self_attention_cyber = SelfAttention(embed_dim=128, num_heads=attention_heads)
        self.self_attention_physical = SelfAttention(embed_dim=128, num_heads=attention_heads)

        # Transformer Encoder Layers for both sides
        self.transformer_encoder_cyber = TransformerEncoder(embed_dim=128, num_heads=attention_heads, num_layers=2)
        self.transformer_encoder_physical = TransformerEncoder(embed_dim=128, num_heads=attention_heads, num_layers=2)

        # Cross Attention
        self.cross_attention = CrossAttention(embed_dim=128, num_heads=attention_heads)

        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            KANLinear(128 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            self.dropout,
            KANLinear(256, num_classes)
        )

    def forward(self, x_cyber, x_physical):
        # Cyber branch
        x_cyber = self.conv1_cyber(x_cyber.unsqueeze(1))
        x_cyber = self.pyramid_cyber(x_cyber)
        x_cyber = self.connect_cyber(x_cyber)

        # Prepare for Cyber transformer: (batch, seq_len, features)
        x_cyber = x_cyber.transpose(1, 2)
        x_cyber = self.positional_encoding_cyber(x_cyber)
        x_cyber = self.self_attention_cyber(x_cyber)
        x_cyber = self.transformer_encoder_cyber(x_cyber)

        # Physical branch
        x_physical = self.conv1_physical(x_physical.unsqueeze(1))
        x_physical = self.dilated_physical(x_physical)
        x_physical = self.connect_physical(x_physical)

        # Prepare for Physical transformer: (batch, seq_len, features)
        x_physical = x_physical.transpose(1, 2)
        x_physical = self.positional_encoding_physical(x_physical)
        x_physical = self.self_attention_physical(x_physical)
        x_physical = self.transformer_encoder_physical(x_physical)

        # Cross Attention between Cyber and Physical
        x_cyber_attn = self.cross_attention(x_cyber, x_physical)
        x_physical_attn = self.cross_attention(x_physical, x_cyber)

        # Pooling
        x_cyber_pooled = torch.mean(x_cyber_attn, dim=1)
        x_physical_pooled = torch.mean(x_physical_attn, dim=1)

        # Concatenate
        x_concat = torch.cat((x_cyber_pooled, x_physical_pooled), dim=1)

        # Fully connected layers
        output = self.fc(x_concat)
        return output
