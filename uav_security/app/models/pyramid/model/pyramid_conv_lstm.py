import torch
import torch.nn as nn

from app.models.pyramid.blocks.cross_attn import CrossAttention
from app.models.pyramid.blocks.dilated_conv import DilatedConv1D
from app.models.pyramid.blocks.pyramid_conv import PyramidConv1D





# Updated Model with Depth and Cross Attention
class PyramidConvLSTM(nn.Module):
    def __init__(self, input_dim_cyber, input_dim_physical, num_classes, num_layers=3, lstm_layers=2, attention_heads=4):
        super(PyramidConvLSTM, self).__init__()
        self.in_channels_cyber = 64
        self.in_channels_physical = 64

        # Cyber branch with Deeper Pyramid Convolution
        self.conv1_cyber = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pyramid_cyber = PyramidConv1D(64, 128, num_layers=num_layers)
        self.connect_cyber = nn.Conv1d(128, 64, kernel_size=1)
        self.lstm_cyber = nn.LSTM(input_size=64, hidden_size=128, num_layers=lstm_layers, batch_first=True)

        # Physical branch with Deeper Dilated Convolution
        self.conv1_physical = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.dilated_physical = DilatedConv1D(64, 128, dilation=2, num_layers=num_layers)
        self.connect_physical = nn.Conv1d(128, 64, kernel_size=1)
        self.lstm_physical = nn.LSTM(input_size=64, hidden_size=128, num_layers=lstm_layers, batch_first=True)

        # Cross Attention
        self.cross_attention = CrossAttention(embed_dim=128, num_heads=attention_heads)

        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_cyber, x_physical):
        # Cyber branch
        x_cyber = self.conv1_cyber(x_cyber.unsqueeze(1))
        x_cyber = self.pyramid_cyber(x_cyber)
        x_cyber = self.connect_cyber(x_cyber)
        x_cyber, _ = self.lstm_cyber(x_cyber.transpose(1, 2))

        # Physical branch
        x_physical = self.conv1_physical(x_physical.unsqueeze(1))
        x_physical = self.dilated_physical(x_physical)
        x_physical = self.connect_physical(x_physical)
        x_physical, _ = self.lstm_physical(x_physical.transpose(1, 2))  # (batch, seq_len, features)

        # Cross Attention between Cyber and Physical
        x_cyber_attn = self.cross_attention(x_cyber, x_physical)
        x_physical_attn = self.cross_attention(x_physical, x_cyber)

        # Concatenate the outputs from both branches after attention
        x_concat = torch.cat((x_cyber_attn[:, -1, :], x_physical_attn[:, -1, :]), dim=1)

        # Fully connected layers
        output = self.fc(x_concat)
        return output
