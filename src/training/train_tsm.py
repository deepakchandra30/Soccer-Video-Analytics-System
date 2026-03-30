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
from src.models.temporal.losses import get_class_weights
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from src.training.trainer import (
    train_epoch, validate_epoch, EarlyStopping, save_checkpoint,
)
from src.evaluation.predict import generate_predictions
from src.evaluation.evaluate import run_evaluation


def load_matches(data_dir, split, feature_type="pca512"):
    """Load all match features and annotations for a split."""
    from SoccerNet.utils import getListGames
    import json

    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_resnet50_2048.npy", "2_resnet50_2048.npy"),
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
        label_path = os.path.join(game_dir, "Labels-v3.json")

        if not all(os.path.exists(p) for p in [f1_path, f2_path, label_path]):
            continue

        half1 = np.load(f1_path)
        half2 = np.load(f2_path)
        features = np.concatenate([half1, half2], axis=0)

        with open(label_path) as f:
            labels = json.load(f)
        actions_raw = labels.get("actions", {})
        if isinstance(actions_raw, dict):
            annotations = [
                v["imageMetadata"]
                for v in actions_raw.values()
                if isinstance(v, dict) and "imageMetadata" in v
            ]
        else:
            annotations = actions_raw

        matches.append((features, annotations))
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
    parser.add_argument("--feat-dim", type=int, default=TSM_CONFIG["feat_dim"])
    parser.add_argument("--feature-type", default="pca512",
                        choices=["pca512", "resnet50"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best.pt and run evaluation only")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.feature_type} features from {args.data_dir}...")
    if args.eval_only:
        val_matches, val_dirs, val_ids = load_matches(args.data_dir, "valid", args.feature_type)
        train_matches = []
    else:
        train_matches, _, _ = load_matches(args.data_dir, "train", args.feature_type)
        val_matches, val_dirs, val_ids = load_matches(args.data_dir, "valid", args.feature_type)
    print(f"  train: {len(train_matches)} matches, valid: {len(val_matches)} matches")

    model = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=TSM_CONFIG["hidden_dim"], n_shifts=TSM_CONFIG["n_shifts"],
    ).to(args.device)

    if args.eval_only:
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    import wandb
    wandb.init(project=args.wandb_project, name="tsm-baseline",
               config={**TSM_CONFIG, "feat_dim": args.feat_dim,
                       "feature_type": args.feature_type})

    if not args.eval_only:
        train_ds = ChunkedSoccerNetDataset(
            train_matches, chunk_size=TSM_CONFIG["chunk_size"],
            event_ratio=TSM_CONFIG["event_ratio"], feat_dim=args.feat_dim,
        )
        val_ds = ChunkedSoccerNetDataset(
            val_matches, chunk_size=TSM_CONFIG["chunk_size"],
            event_ratio=0.0, feat_dim=args.feat_dim,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=TSM_CONFIG["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        weights = get_class_weights(TSM_CONFIG["num_classes"], TSM_CONFIG["bg_weight"])
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))
        early_stop = EarlyStopping(patience=TSM_CONFIG["patience"], mode="min")

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

    print("Generating predictions on validation split...")
    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_resnet50_2048.npy", "2_resnet50_2048.npy"),
    }
    pred_dir = os.path.join(args.output_dir, "predictions")
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=pred_dir, feature_files=feat_map[args.feature_type],
        window_size=TSM_CONFIG["window_size"], stride=TSM_CONFIG["stride"],
        nms_window=TSM_CONFIG["nms_window"],
        confidence_threshold=TSM_CONFIG["confidence_threshold"],
        framerate=TSM_CONFIG["framerate"], device=args.device,
    )

    print("Running SoccerNet evaluation...")
    results = run_evaluation(args.data_dir, pred_dir, split="valid")
    avg_map = results.get("a_mAP", 0.0)

    map_1s = results.get("a_mAP_per_class_at1", results.get("a_mAP_at1", None))
    map_2s = results.get("a_mAP_per_class_at2", results.get("a_mAP_at2", None))
    map_5s = results.get("a_mAP_per_class_at5", results.get("a_mAP_at5", None))

    print(f"\nmAP Results:")
    print(f"  avg-mAP tight: {avg_map:.1f}%")
    if map_1s is not None:
        print(f"  mAP@1s:        {map_1s:.1f}%")
    if map_2s is not None:
        print(f"  mAP@2s:        {map_2s:.1f}%")
    if map_5s is not None:
        print(f"  mAP@5s:        {map_5s:.1f}%")

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
