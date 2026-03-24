"""Reusable building blocks for temporal classification heads."""
import torch.nn as nn


class Conv1dBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + Dropout."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
