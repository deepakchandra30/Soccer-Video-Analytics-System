"""Training utilities for temporal action spotting."""
import os

import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            logits = model(features)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

    return total_loss / max(1, len(loader))


def train_epoch_downsampled(model, loader, optimizer, criterion, device, target_stride=4):
    """Train one epoch for models whose temporal output is downsampled vs targets."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        features = batch["features"].to(device)   # (B, T, C)
        targets = batch["targets"].to(device)     # (B, T)

        optimizer.zero_grad()
        logits = model(features)                  # (B, T_out, K)

        # Downsample targets to match model temporal resolution
        targets_ds = targets[:, ::target_stride]  # nominally (B, T/stride)

        # Safety for any edge mismatch
        t_out = logits.size(1)
        t_tgt = targets_ds.size(1)
        t_min = min(t_out, t_tgt)
        logits = logits[:, :t_min, :]
        targets_ds = targets_ds[:, :t_min]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets_ds.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def validate_epoch_downsampled(model, loader, criterion, device, target_stride=4):
    """Validate one epoch for models whose temporal output is downsampled vs targets."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            features = batch["features"].to(device)   # (B, T, C)
            targets = batch["targets"].to(device)     # (B, T)

            logits = model(features)                  # (B, T_out, K)
            targets_ds = targets[:, ::target_stride]  # (B, T/stride)

            t_out = logits.size(1)
            t_tgt = targets_ds.size(1)
            t_min = min(t_out, t_tgt)
            logits = logits[:, :t_min, :]
            targets_ds = targets_ds[:, :t_min]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets_ds.reshape(-1),
            )
            total_loss += loss.item()

    return total_loss / max(1, len(loader))


class EarlyStopping:
    """Stop training when validation metric stops improving."""
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" \
            else (value > self.best + self.min_delta)

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)