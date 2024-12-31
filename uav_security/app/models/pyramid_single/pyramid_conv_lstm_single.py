import torch
import torch.nn as nn
from app.model.pyramid.blocks.pyramid_conv import PyramidConv1D

# Updated PyramidConvLSTM Model
class PyramidConvLSTM(nn.Module):
    """
    Pyramid Convolutional LSTM Model with Single Input.
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_layers: int = 6,
                 lstm_hidden_dim: int = 512,
                 dropout: float = 0.5):
        '''
        :param input_dim: Input feature dimension
        :param num_classes: Number of classes to classify
        :param num_layers: Number of pyramid conv layers
        :param lstm_hidden_dim: Hidden dimension of LSTM
        :param dropout: Dropout rate
        '''
        super(PyramidConvLSTM, self).__init__()
        self.in_channels = 128  # Increased from 64 to 128

        self.dropout = nn.Dropout(dropout)

        # Input convolution with Pyramid Convolution
        self.conv1 = nn.Conv1d(1, 128, kernel_size=7, stride=2, padding=3)  # Increased output channels
        self.pyramid_conv = PyramidConv1D(128, 256, num_layers=num_layers)  # Increased channels

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True)  # Increased hidden size and added bidirectionality

        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 512),  # Multiply by 2 due to bidirectional
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, seq_len)
        x = self.conv1(x.unsqueeze(1))  # (batch, 1, seq_len) -> (batch, 128, seq_len/2)
        x = self.pyramid_conv(x)  # (batch, 256, seq_len/2)

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len/2, 256)
        x, _ = self.lstm(x)    # LSTM output: (batch, seq_len/2, lstm_hidden_dim * 2)

        # Pooling
        x_pooled = torch.mean(x, dim=1)  # (batch, lstm_hidden_dim * 2)

        # Fully connected layer
        output = self.fc(x_pooled)
        return output
