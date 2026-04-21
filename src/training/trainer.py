"""Shared training utilities for temporal action spotting models."""
import torch


def _align_targets(logits, targets):
    """Downsample targets when the model emits at a lower temporal resolution.

    SlowFast outputs (B, T // slow_stride, C+1) by design, while
    ChunkedSoccerNetDataset yields per-frame targets at full T. Without this
    alignment cross_entropy crashes with a batch_size mismatch (T_slow*B vs
    T*B). Keeping the stride-awareness here means the trainer stays generic
    and SlowFast doesn't have to fake full resolution.
    """
    if logits.dim() == 3 and targets.dim() == 2 and logits.size(1) != targets.size(1):
        if targets.size(1) % logits.size(1) == 0:
            stride = targets.size(1) // logits.size(1)
            targets = targets[:, ::stride]
    return targets


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Standard training loop. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        logits = model(features)  # (B, T_out, C); T_out may be < T for SlowFast
        targets = _align_targets(logits, targets)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(model, dataloader, criterion, device):
    """Validation pass. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            logits = model(features)
            targets = _align_targets(logits, targets)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


class EarlyStopping:
    """Stop training when a metric stops improving."""
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric):
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, epoch, metric, path):
    """Save model checkpoint with training metadata."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Restore model (and optionally optimizer) from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metric", 0)
