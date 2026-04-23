"""TSM training entry point for SoccerNet action spotting."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.seeds import set_seeds
from config.tsm_config import TSM_CONFIG
from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.losses import (
    get_class_weights, compute_class_weights_from_matches,
)
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from src.training.trainer import (
    train_epoch, validate_epoch, EarlyStopping, save_checkpoint,
)
from src.evaluation.predict import generate_predictions
from src.evaluation.evaluate import run_evaluation


# Per feature type, multiplicative scale applied on read to keep the
# network's input std near 1.0 — crucial for gradient flow on tiny-magnitude
# feature sets. PCA-512 is already ~N(0,1) so scale stays 1.0; Baidu raw
# post-ReLU features come in at std≈0.073 which starves gradient flow and
# produced the 2.39% mAP collapse on 2026-04-22. Scaling by 1/std≈13.7
# (rounded up to 15 for safety) restores trainable signal.
FEATURE_SCALE = {
    "pca512": 1.0,
    "resnet50": 1.0,
    "baidu": 15.0,
}

# SoccerNet features ship at different framerates depending on who extracted
# them. PCA-512 and raw ResNet-2048 are 2 fps. Baidu embeddings are 1 fps
# (half-1 is ~2700 frames for a 45-min half = 1.0 fps). Annotation→frame
# mapping must use the matching rate or events land at the wrong indices
# (2x offset for Baidu), silently destroying training signal — the 2.39%
# mAP regression on 2026-04-22 was traced to this mismatch.
FEATURE_FRAMERATE = {
    "pca512": 2,
    "resnet50": 2,
    "baidu": 1,
}


class _HalfConcatView:
    """Zero-copy lazy concat of two half-feature memory-maps with optional
    per-read scalar scaling.

    Quacks like ``np.concatenate([h1, h2], axis=0) * scale`` for the access
    patterns ChunkedSoccerNetDataset needs (``.shape`` and slice/int
    ``__getitem__``), but never materializes the full (n1+n2) array.
    Slicing reads only the touched rows through the mmap, so peak RAM
    scales with chunk_size, not total frames. Essential for 8576-dim Baidu
    features where eagerly concatenating 800 matches overruns 23 GB RAM.
    """
    def __init__(self, h1, h2, scale=1.0):
        assert h1.shape[1:] == h2.shape[1:], "half feature shapes must match"
        self._h1 = h1
        self._h2 = h2
        self._n1 = h1.shape[0]
        self._scale = float(scale)
        self.shape = (self._n1 + h2.shape[0], *h1.shape[1:])
        self.dtype = h1.dtype
        self.ndim = h1.ndim

    def __len__(self):
        return self.shape[0]

    def _scaled(self, arr):
        if self._scale == 1.0:
            return np.ascontiguousarray(arr)
        return (np.asarray(arr, dtype=np.float32) * self._scale)

    def __getitem__(self, key):
        n1 = self._n1
        total = self.shape[0]
        if isinstance(key, slice):
            start, stop, step = key.indices(total)
            if step != 1:
                full = np.concatenate([np.asarray(self._h1), np.asarray(self._h2)])
                return self._scaled(full[key])
            if stop <= n1:
                return self._scaled(self._h1[start:stop])
            if start >= n1:
                return self._scaled(self._h2[start - n1:stop - n1])
            a = self._scaled(self._h1[start:n1])
            b = self._scaled(self._h2[0:stop - n1])
            return np.concatenate([a, b], axis=0)
        if isinstance(key, (int, np.integer)):
            if key < 0:
                key += total
            if not (0 <= key < total):
                raise IndexError(key)
            return self._scaled(self._h1[key] if key < n1 else self._h2[key - n1])
        full = np.concatenate([np.asarray(self._h1), np.asarray(self._h2)])
        return self._scaled(full[key])


def load_matches(data_dir, split, feature_type="pca512"):
    """Load all match features and annotations for a split.

    Prefers Labels-v2.json because it contains the full action spotting
    label set (~200 annotations/game).  Labels-v3.json's ``actions`` field
    only holds the subset of frames selected for the bbox/replay tasks
    (~15/game locally), so training on it starves the model of positive
    examples and sinks avg-mAP.  Labels-v3 is kept as a last-resort
    fallback only so fixtures without v2 still load.

    Features are loaded with ``mmap_mode='r'`` and wrapped in
    ``_HalfConcatView`` so peak memory is O(chunk_size * batch_size) rather
    than O(total_frames). Required for 8576-dim Baidu features.
    """
    from SoccerNet.utils import getListGames
    import json

    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_ResNET_TF2.npy", "2_ResNET_TF2.npy"),
        "baidu": ("1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"),
    }
    f1_name, f2_name = feat_map[feature_type]
    games = getListGames(split=split)
    matches = []
    game_dirs = []
    game_ids = []

    for game in games:
        game_dir = os.path.join(data_dir, game)
        f1_path = os.path.join(game_dir, f1_name)
        f2_path = os.path.join(game_dir, f2_name)

        if not all(os.path.exists(p) for p in [f1_path, f2_path]):
            continue

        # Prefer Labels-v2 (full action-spotting set); fall back to v3.
        label_v2 = os.path.join(game_dir, "Labels-v2.json")
        label_v3 = os.path.join(game_dir, "Labels-v3.json")

        if os.path.exists(label_v2):
            label_path = label_v2
        elif os.path.exists(label_v3):
            label_path = label_v3
        else:
            continue

        half1 = np.load(f1_path, mmap_mode="r")
        half2 = np.load(f2_path, mmap_mode="r")
        features = _HalfConcatView(
            half1, half2, scale=FEATURE_SCALE.get(feature_type, 1.0),
        )
        half1_len = half1.shape[0]

        with open(label_path) as f:
            labels = json.load(f)

        # Labels-v3.json: ``actions`` is a dict {key: {imageMetadata: {...}}}
        # Labels-v2.json: ``annotations`` is a list of dicts with gameTime/label
        if "actions" in labels:
            actions = labels["actions"]
            if isinstance(actions, dict):
                annotations = [v.get("imageMetadata", {}) for v in actions.values()]
            else:
                annotations = list(actions)
        else:
            annotations = labels.get("annotations", [])

        # Returning half1_len is load-bearing: ChunkedSoccerNetDataset needs it
        # to offset half-2 annotations to the correct index in the concatenated
        # feature array. Without it, ~60% of events land at the wrong frame.
        matches.append((features, annotations, half1_len))
        game_dirs.append(game_dir)
        game_ids.append(game)

    return matches, game_dirs, game_ids


def collate_fn(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train TSM action spotting model")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/tsm_baseline")
    parser.add_argument("--epochs", type=int, default=TSM_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TSM_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TSM_CONFIG["lr"])
    parser.add_argument("--device", default=None)
    # feat-dim defaults to None so it can auto-derive from feature-type below.
    # Users who want a custom dim (e.g. half-precision experiments) still pass
    # --feat-dim explicitly.
    parser.add_argument("--feat-dim", type=int, default=None)
    parser.add_argument("--feature-type", default="pca512",
                        choices=["pca512", "resnet50", "baidu"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    # Submission-only: merge an extra labelled split into training data.
    # Default None preserves existing train-on-train-only behaviour exactly.
    parser.add_argument("--extra-train-split", default=None,
                        choices=[None, "valid", "test"],
                        help="Optionally merge another split into training. Use "
                             "'valid' for challenge-submission retrains; the "
                             "loader will then validate on 'test' so val_loss "
                             "still drives early stopping without leakage from "
                             "the new training data.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Auto-derive feat_dim from feature_type when not overridden. This
    # lets --feature-type baidu "just work" without also passing
    # --feat-dim 8576 every time.
    if args.feat_dim is None:
        _FEAT_DIMS = {"pca512": 512, "resnet50": 2048, "baidu": 8576}
        args.feat_dim = _FEAT_DIMS[args.feature_type]

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    print(f"Loading {args.feature_type} features from {args.data_dir}...")
    train_matches, _, _ = load_matches(args.data_dir, "train", args.feature_type)
    # Submission retrains merge another split into training data. Pick a
    # validation split that's disjoint from the training pool so val_loss
    # still means something.
    val_split_name = "test" if args.extra_train_split == "valid" else "valid"
    val_matches, val_dirs, val_ids = load_matches(
        args.data_dir, val_split_name, args.feature_type,
    )
    if args.extra_train_split:
        extra_matches, _, _ = load_matches(
            args.data_dir, args.extra_train_split, args.feature_type,
        )
        train_matches = train_matches + extra_matches
        print(f"  merged extra split '{args.extra_train_split}' "
              f"({len(extra_matches)} matches) into training data.")
    print(f"  train: {len(train_matches)} matches, "
          f"valid-on-{val_split_name}: {len(val_matches)} matches")

    # datasets and loaders — framerate MUST match the feature extraction rate
    # or annotations land at wrong frame indices (see FEATURE_FRAMERATE note).
    _framerate = FEATURE_FRAMERATE.get(args.feature_type, 2)
    train_ds = ChunkedSoccerNetDataset(
        train_matches, chunk_size=TSM_CONFIG["chunk_size"],
        event_ratio=TSM_CONFIG["event_ratio"], feat_dim=args.feat_dim,
        framerate=_framerate,
    )
    val_ds = ChunkedSoccerNetDataset(
        val_matches, chunk_size=TSM_CONFIG["chunk_size"],
        event_ratio=0.0, feat_dim=args.feat_dim,
        framerate=_framerate,
    )
    # num_workers>0 parallelises mmap reads; critical for Baidu (8576-dim)
    # where each 40-frame chunk is 1.3 MB and the sync loader is I/O-bound.
    # mmap pages are shared across fork()ed workers, so no extra RAM.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    # model, optimizer, scheduler, loss
    model = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=TSM_CONFIG["hidden_dim"], n_shifts=TSM_CONFIG["n_shifts"],
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=TSM_CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Use per-class inverse-frequency weights derived from the actual train
    # annotations. Uniform weights were undercounting rare classes (red card,
    # penalty) during tight-mAP evaluation — each of those classes gets
    # averaged into the final avg-mAP at weight 1/17, so a single zero-AP
    # class costs ~5 points on its own.
    weights = compute_class_weights_from_matches(
        train_matches, num_classes=TSM_CONFIG["num_classes"],
        bg_weight=TSM_CONFIG["bg_weight"],
    )
    print(f"Per-class weights: bg={weights[0]:.3f}, "
          f"events min={weights[1:].min():.2f} max={weights[1:].max():.2f}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))

    early_stop = EarlyStopping(patience=TSM_CONFIG["patience"], mode="min")

    # wandb
    import wandb
    wandb.init(project=args.wandb_project, name="tsm-baseline",
               config={**TSM_CONFIG, "feat_dim": args.feat_dim,
                       "feature_type": args.feature_type})

    # training loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = validate_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        wandb.log({"train/loss": train_loss, "val/loss": val_loss,
                    "lr": scheduler.get_last_lr()[0], "epoch": epoch})
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, "best.pt"))

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # generate predictions on validation set
    print("Generating predictions on validation split...")
    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_ResNET_TF2.npy", "2_ResNET_TF2.npy"),
        "baidu": ("1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"),
    }
    pred_dir = os.path.join(args.output_dir, "predictions")
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=pred_dir, feature_files=feat_map[args.feature_type],
        window_size=TSM_CONFIG["window_size"], stride=TSM_CONFIG["stride"],
        nms_window=TSM_CONFIG["nms_window"],
        confidence_threshold=TSM_CONFIG["confidence_threshold"],
        framerate=_framerate, device=args.device,
        feature_scale=FEATURE_SCALE.get(args.feature_type, 1.0),
    )

    # evaluate -- run_evaluation stubs missing predictions and skips
    # games without Labels-v2.json so the SDK never hits FileNotFoundError.
    print("Running SoccerNet evaluation...")
    results = run_evaluation(args.data_dir, pred_dir, split=val_split_name)

    # SoccerNet's evaluator returns numpy scalars / arrays in [0, 1].
    # The previous display of 0.0% was a formatting bug: the raw value
    # 0.0047 (= 0.47%) was being passed to {:.1f}% which rounds to 0.0.
    # Always coerce to float and multiply by 100 to get a percentage.
    def _as_pct(v):
        if v is None:
            return None
        try:
            return float(v) * 100.0
        except (TypeError, ValueError):
            return None

    avg_map = _as_pct(results.get("a_mAP", 0.0))
    map_1s = _as_pct(results.get("a_mAP_per_class_at1", results.get("a_mAP_at1", None)))
    map_2s = _as_pct(results.get("a_mAP_per_class_at2", results.get("a_mAP_at2", None)))
    map_5s = _as_pct(results.get("a_mAP_per_class_at5", results.get("a_mAP_at5", None)))

    print(f"\nmAP Results:")
    print(f"  avg-mAP tight: {avg_map:.4f}%")
    if map_1s is not None:
        print(f"  mAP@1s:        {map_1s:.4f}%")
    if map_2s is not None:
        print(f"  mAP@2s:        {map_2s:.4f}%")
    if map_5s is not None:
        print(f"  mAP@5s:        {map_5s:.4f}%")

    wandb.log({"eval/avg_mAP_tight": avg_map})
    if map_1s is not None:
        wandb.log({"eval/mAP_1s": map_1s})
    if map_2s is not None:
        wandb.log({"eval/mAP_2s": map_2s})
    if map_5s is not None:
        wandb.log({"eval/mAP_5s": map_5s})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
