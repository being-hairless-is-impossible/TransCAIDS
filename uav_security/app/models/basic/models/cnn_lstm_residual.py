import torch
import torch.nn as nn

class CNNLSTMResidual(nn.Module):
    torch.autograd.set_detect_anomaly(True)

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(CNNLSTMResidual, self).__init__()

        # First conv block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()  # No inplace=True

        # Second conv block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # FC
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Residual connections
        self.residual_conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1)  # Match dimensions for residual
        self.residual_conv2 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1)  # Match dimensions

    def forward(self, x):
        # Residual input (before passing through first conv block)
        residual1 = self.residual_conv1(x.unsqueeze(1))  # Add channel dimension

        # First conv block
        x = self.conv1(x.unsqueeze(1))
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)

        # Resize residual1 to match x before adding
        if residual1.size(2) != x.size(2):
            residual1 = nn.functional.interpolate(residual1, size=x.size(2))  # Adjust size of residual

        # Residual connection after the first conv block
        x = x + residual1  # Out-of-place addition

        # Second conv block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.relu(x)

        # Residual connection after second conv block
        residual2 = self.residual_conv2(residual1)
        if residual2.size(2) != x.size(2):
            residual2 = nn.functional.interpolate(residual2, size=x.size(2))  # Adjust size of residual

        x = x + residual2  # Out-of-place addition

        # Prepare for LSTM, transpose to (batch_size, seq_len, feature_dim)
        x = x.transpose(1, 2)

        # LSTM part
        out, _ = self.lstm(x)

        # Take the last hidden state
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return out
