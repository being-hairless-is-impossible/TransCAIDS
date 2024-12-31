from torch import nn


class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, num_layers=3):
        super(DilatedConv1D, self).__init__()
        self.layers = nn.ModuleList()
        current_in_channels = in_channels  # Initialize with the initial in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(current_in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation))
            self.layers.append(nn.BatchNorm1d(out_channels))
            current_in_channels = out_channels  # Update in_channels for the next iteration
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x
