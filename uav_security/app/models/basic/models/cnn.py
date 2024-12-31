import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(CNN, self).__init__()

        # The hidden_dim and num_layers are accepted for compatibility but not used
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.input_dim = input_dim

        # Compute the output length after the convolution and pooling layers
        def compute_output_length(input_length, kernel_size, stride, padding):
            return ((input_length + 2 * padding - (kernel_size - 1) - 1) // stride) + 1

        length = input_dim
        # After conv1
        length = compute_output_length(length, kernel_size=3, stride=1, padding=1)
        # After pool1
        length = compute_output_length(length, kernel_size=2, stride=2, padding=0)
        # After conv2
        length = compute_output_length(length, kernel_size=3, stride=1, padding=1)
        # After pool2
        length = compute_output_length(length, kernel_size=2, stride=2, padding=0)

        self.cnn_output_length = length

        # Fully connected layer
        self.fc = nn.Linear(128 * self.cnn_output_length, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D: (batch_size, 1, input_dim)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # x shape after first conv and pool: (batch_size, 64, cnn_output_length_after_pool1)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # x shape after second conv and pool: (batch_size, 128, cnn_output_length_after_pool2)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128 * cnn_output_length_after_pool2)

        # Fully connected layer
        out = self.fc(x)  # Shape: (batch_size, num_classes)
        return out
