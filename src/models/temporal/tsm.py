"""TSM temporal shift module adapted for pre-extracted feature sequences."""
import torch
import torch.nn as nn
from einops import rearrange


class TemporalShift(nn.Module):
    """Shift 1/n_div channels forward and backward in time.

    Adapted from TSM (Lin et al., ICCV 2019) to work on pre-extracted
    feature vectors instead of CNN feature maps.
    """
    def __init__(self, n_div=8):
        super().__init__()
        self.n_div = n_div

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        fold = C // self.n_div
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]             # shift left (past)
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]  # shift right (future)
        return out


class TSMSpottingHead(nn.Module):
    """TSM + 1D conv/GRU head for action spotting on pre-extracted features.

    Takes (B, T, feat_dim) feature sequences and produces (B, T, num_classes+1)
    per-frame logits. The +1 is for the background class at index 0.
    """
    def __init__(self, feat_dim=512, num_classes=17, hidden_dim=256, n_shifts=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.shifts = nn.ModuleList([TemporalShift() for _ in range(n_shifts)])

        self.conv1 = nn.Conv1d(feat_dim, hidden_dim, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, features):
        # features: (B, T, feat_dim)
        if features.shape[-1] != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim={self.feat_dim}, got {features.shape[-1]}"
            )

        x = features
        for shift in self.shifts:
            x = shift(x)

        # conv layers expect (B, C, T)
        x = rearrange(x, 'b t c -> b c t')
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))

        # GRU expects (B, T, C)
        x = rearrange(x, 'b c t -> b t c')
        x, _ = self.gru(x)
        x = self.dropout(x)

        return self.fc(x)  # (B, T, num_classes+1)
