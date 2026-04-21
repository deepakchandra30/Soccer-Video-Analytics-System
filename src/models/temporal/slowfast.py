"""SlowFast dual-pathway model for action spotting on pre-extracted features.

Adapted from Feichtenhofer et al. (ICCV 2019). Instead of processing raw
video at two framerates, we subsample pre-extracted 2fps features at
different strides to create slow (context) and fast (detail) streams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlowFastSpotting(nn.Module):
    """Dual-stride temporal model for action spotting.

    Slow pathway: subsampled at slow_stride for cheap large-context modelling,
        then upsampled back to T so fusion and output run at full resolution.
    Fast pathway: full temporal resolution, lightweight (hidden_dim // 4).
    Lateral: project fast -> hidden_dim; fuse with slow at full T.

    Output: logits at **full T** (num_classes + 1). Previously the model
    produced logits at T/slow_stride (e.g. T/4 at 2fps -> effective 0.5fps).
    With `stride=4` and tight mAP evaluation at 1-5s tolerances, that coarse
    output caused the upstream `sliding_window_inference` to linear-interp
    predictions and smear event peaks across ~2s, capping SlowFast's tight-mAP
    below TSM's. Outputting at full T removes that ceiling.
    """
    def __init__(self, feat_dim=512, num_classes=17, slow_stride=4,
                 hidden_dim=256):
        super().__init__()
        self.slow_stride = slow_stride
        fast_dim = hidden_dim // 4

        # slow pathway
        self.slow_proj = nn.Linear(feat_dim, hidden_dim)
        self.slow_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # fast pathway
        self.fast_proj = nn.Linear(feat_dim, fast_dim)
        self.fast_conv = nn.Sequential(
            nn.Conv1d(fast_dim, fast_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(fast_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # lateral: project fast pathway to hidden_dim at full T
        self.lateral = nn.Conv1d(fast_dim, hidden_dim, kernel_size=1)

        # classification head
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, features):
        # features: (B, T, feat_dim)
        B, T, C = features.shape

        # slow pathway: subsample -> process -> upsample back to T
        slow_in = features[:, ::self.slow_stride, :]
        slow_x = self.slow_proj(slow_in)
        slow_x = rearrange(slow_x, 'b t c -> b c t')
        slow_x = self.slow_conv(slow_x)
        # upsample slow output to full T so fusion + classification head run
        # at the same resolution as tight-mAP evaluation expects
        if slow_x.size(2) != T:
            slow_x = F.interpolate(slow_x, size=T, mode='linear',
                                   align_corners=False)

        # fast pathway: full resolution
        fast_x = self.fast_proj(features)
        fast_x = rearrange(fast_x, 'b t c -> b c t')
        fast_x = self.fast_conv(fast_x)

        # lateral: project fast to hidden_dim at full T (no downsampling)
        fast_lateral = self.lateral(fast_x)

        # fuse slow + fast at full T
        fused = torch.cat([slow_x, fast_lateral], dim=1)
        fused = rearrange(fused, 'b c t -> b t c')

        # GRU + classification (output at full T, not T/slow_stride)
        fused, _ = self.gru(fused)
        fused = self.dropout(fused)
        return self.fc(fused)
