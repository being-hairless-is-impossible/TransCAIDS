import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(CNNLSTM, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # LSTM layers
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # CNN part
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Prepare for LSTM, transpose to (batch_size, seq_len, feature_dim)
        x = x.transpose(1, 2)

        # LSTM part
        out, _ = self.lstm(x)

        # Take the last hidden state
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return out
