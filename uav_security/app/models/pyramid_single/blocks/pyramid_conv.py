from torch import nn
import torch
import torch.nn as nn
from app.models.pyramid.blocks.pyramid_conv import PyramidConv1D


class PyramidConv(nn.Module):
    """
    Pyramid Convolutional Model for Single Input.
    """
    def __init__(self, input_dim, num_classes, num_layers=3, lstm_layers=2):
        super(PyramidConv, self).__init__()
        self.in_channels = 64

        # Input convolution with Pyramid Convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pyramid_conv = PyramidConv1D(64, 128, num_layers=num_layers)
        self.connect_conv = nn.Conv1d(128, 64, kernel_size=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=lstm_layers, batch_first=True)

        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.conv1(x.unsqueeze(1))  # (batch, 1, seq_len) -> (batch, 64, seq_len/2)
        x = self.pyramid_conv(x)
        x = self.connect_conv(x)  # (batch, 64, seq_len/2)
        x = x.transpose(1, 2)  # (batch, seq_len/2, 64)
        x, _ = self.lstm(x)     # (batch, seq_len/2, 128)

        # Use the last time step's output
        x_last = x[:, -1, :]    # (batch, 128)

        # Fully connected layers
        output = self.fc(x_last)
        return output
