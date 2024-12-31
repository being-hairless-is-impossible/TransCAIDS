from torch import nn


class PyramidConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super(PyramidConv1D, self).__init__()

        self.layers = nn.ModuleList()
        current_in_channels = in_channels
        for _ in range(num_layers):
            # First Conv1d layer in the iteration
            self.layers.append(nn.Conv1d(current_in_channels, out_channels, kernel_size=3, padding=1))
            current_in_channels = out_channels  # Update in_channels to out_channels
            # Subsequent Conv1d layers
            self.layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2))
            self.layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3))
            self.layers.append(nn.BatchNorm1d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x