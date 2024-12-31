import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is of shape (batch_size, input_size)

        # Reshape x to (batch_size, seq_length, input_size)
        # Since each sample is a single time step, set seq_length = 1
        x = x.unsqueeze(1)  # Now x is (batch_size, 1, input_size)

        batch_size = x.size(0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Since seq_length = 1, out is of shape (batch_size, 1, hidden_dim)
        out = out[:, -1, :]  # Get the output of the last time step

        # Fully connected layer
        out = self.fc(out)
        return out



