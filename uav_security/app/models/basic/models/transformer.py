import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Embedding layer to project input features to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=1)

        # Transformer Encoder with pre-layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-Layer Normalization
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights using Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.embedding(x)  # x shape: (batch_size, d_model)
        x = x.unsqueeze(1)     # x shape: (batch_size, seq_length=1, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # x shape: (batch_size, seq_length=1, d_model)
        x = x.squeeze(1)  # x shape: (batch_size, d_model)
        out = self.fc(x)  # out shape: (batch_size, num_classes)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Since max_len=1, positional encoding is minimal
        pe = torch.zeros(max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
