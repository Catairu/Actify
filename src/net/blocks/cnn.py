import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout=0.5
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=2)  # 95.55

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x
