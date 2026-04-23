"""NetVLAD-based per-frame spotting head (additive — no edits to existing models).

Complements the TSM and SlowFast ensembles by providing a **context
aggregation** pattern rather than local temporal convolution (TSM) or
multi-timescale sampling (SlowFast). The three patterns capture
qualitatively different temporal structure, so their ensemble errors
decorrelate.

Architecture:
    1. Local 1D conv over pre-extracted features to a common hidden dim.
    2. NetVLAD pooling over the whole window: K learnable cluster centres,
       soft-assigned residuals, L2-intra-normalise + L2-normalise.
    3. Per-frame classifier that concatenates each frame's projected
       feature with the broadcast-global NetVLAD vector. This preserves
       the per-frame output resolution that the SoccerNet evaluator (and
       our ensemble's sliding_window_inference) needs — (B, T, C+1).

Why per-frame + global context instead of pure per-frame NetVLAD:
    True per-frame NetVLAD (one VLAD per t using a ±W window) is
    O(T * K * D) per window — too expensive on an RTX 2050. A single
    per-window VLAD broadcast across frames captures the "which cluster
    structures are active in this window?" signal at per-frame cost.

Reference: Giancola & Ghanem 2021 (NetVLAD++ for action spotting).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLADPool(nn.Module):
    """Learnable NetVLAD pooling: (B, T, D) -> (B, K*D).

    Soft-assignment version (as in the original paper): a 1x1 conv
    produces a K-way soft assignment per frame, residuals to K learnable
    cluster centres are weighted and summed, then intra-normalised and
    L2-normalised. The output is suitable for feeding a classifier.
    """
    def __init__(self, feat_dim, num_clusters=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_clusters = num_clusters
        self.assign = nn.Conv1d(feat_dim, num_clusters, kernel_size=1)
        # small init so the softmax is roughly uniform at start -- prevents
        # one cluster from dominating before training has begun.
        self.centers = nn.Parameter(torch.randn(num_clusters, feat_dim) * 0.01)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        x_bct = x.transpose(1, 2)                      # (B, D, T)
        a = self.assign(x_bct).transpose(1, 2)         # (B, T, K)
        a = F.softmax(a, dim=-1)                       # soft assignment

        # Residuals (B, T, K, D) — expand x and centres for broadcasting.
        #   x.unsqueeze(2):               (B, T, 1, D)
        #   centers.unsqueeze(0).unsqueeze(0): (1, 1, K, D)
        residuals = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        weighted = a.unsqueeze(-1) * residuals         # (B, T, K, D)
        vlad = weighted.sum(dim=1)                      # (B, K, D)

        # Intra-normalise each cluster vector, then flatten + L2-normalise.
        vlad = F.normalize(vlad, p=2, dim=-1)
        return F.normalize(vlad.reshape(B, -1), p=2, dim=-1)  # (B, K*D)


class NetVLADSpottingHead(nn.Module):
    """Per-frame spotting via local projection + global NetVLAD context.

    For each frame t, predicts class logits using the concatenation of:
      - the frame's locally-projected feature (from a small 1D conv) and
      - the window-level NetVLAD aggregation broadcast to every frame.

    Output is (B, T, num_classes+1), matching TSM/SlowFast so the existing
    sliding_window_inference + multi_scale_inference + NMS path work
    unchanged.
    """
    def __init__(self, feat_dim=512, num_classes=17, hidden_dim=256,
                 num_clusters=32, dropout=0.4):
        super().__init__()
        self.feat_dim = feat_dim

        self.local_conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.vlad = NetVLADPool(hidden_dim, num_clusters=num_clusters)

        # The classifier sees: per-frame local feature (H) + broadcast global
        # NetVLAD (K*H). Total input dim = H * (1 + K).
        classifier_in = hidden_dim * (1 + num_clusters)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes + 1),
        )

    def forward(self, features):
        # features: (B, T, feat_dim)
        if features.shape[-1] != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim={self.feat_dim}, got {features.shape[-1]}"
            )
        B, T, _ = features.shape

        # Local: conv wants (B, D, T) in, returns (B, H, T)
        x = self.local_conv(features.transpose(1, 2)).transpose(1, 2)  # (B, T, H)

        # Global NetVLAD over the full window, broadcast to each frame
        vlad = self.vlad(x)                               # (B, K*H)
        vlad_exp = vlad.unsqueeze(1).expand(-1, T, -1)    # (B, T, K*H)

        combined = torch.cat([x, vlad_exp], dim=-1)       # (B, T, H + K*H)
        return self.classifier(combined)                  # (B, T, num_classes+1)


if __name__ == "__main__":
    # Self-test: exercise the forward pass on representative shapes and
    # make sure the output matches what the SoccerNet evaluator expects
    # (per-frame logits at full T resolution).
    torch.manual_seed(0)
    B, T, D = 2, 40, 512
    x = torch.randn(B, T, D)
    model = NetVLADSpottingHead(feat_dim=D, num_classes=17, hidden_dim=256,
                                num_clusters=32)
    logits = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(logits.shape)}  (expected ({B}, {T}, 18))")
    assert logits.shape == (B, T, 18), f"shape mismatch: got {tuple(logits.shape)}"
    # Gradient flow smoke test
    loss = logits.mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.grad.abs().sum().item() for p in model.parameters()
                 if p.grad is not None)
    print(f"params: {n_params:,}  total |grad|: {n_grad:.4f}")
    assert n_grad > 0, "no gradient flowed — likely a detached tensor somewhere"
    print("OK")
