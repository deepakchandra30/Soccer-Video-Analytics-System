"""NetVLAD++ training entry point (additive — mirrors train_tsm.py shape).

Zero edits to existing training scripts — this module imports ``load_matches``
and ``collate_fn`` from train_tsm.py so dataset loading stays identical to
the other models (same Labels-v2 handling, same half-2 offset fix).

Training recipe matches train_tsm.py deliberately so the resulting
checkpoint drops straight into the 3-way ensemble without needing its
own chunk size, stride, or per-class weight logic.
"""
import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config.seeds import set_seeds
from config.tsm_config import TSM_CONFIG
from src.models.temporal.netvlad import NetVLADSpottingHead
from src.models.temporal.losses import compute_class_weights_from_matches
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from src.training.trainer import (
    train_epoch, validate_epoch, EarlyStopping, save_checkpoint,
)
from src.training.train_tsm import load_matches, collate_fn
from src.evaluation.predict import generate_predictions
from src.evaluation.evaluate import run_evaluation


# NetVLAD-specific defaults. Reuse TSM_CONFIG for everything that doesn't
# need to differ so we don't introduce a new config module (additive only).
NETVLAD_DEFAULTS = {
    "num_clusters": 32,
    "hidden_dim": 256,
    "dropout": 0.4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train NetVLAD++ spotting head (3rd ensemble model)"
    )
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/netvlad")
    parser.add_argument("--epochs", type=int, default=TSM_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TSM_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TSM_CONFIG["lr"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--feat-dim", type=int, default=None)
    parser.add_argument("--feature-type", default="pca512",
                        choices=["pca512", "resnet50", "baidu"])
    parser.add_argument("--num-clusters", type=int,
                        default=NETVLAD_DEFAULTS["num_clusters"])
    parser.add_argument("--hidden-dim", type=int,
                        default=NETVLAD_DEFAULTS["hidden_dim"])
    parser.add_argument("--dropout", type=float,
                        default=NETVLAD_DEFAULTS["dropout"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    # Submission-only knob, same as train_tsm.py / train_slowfast.py.
    parser.add_argument("--extra-train-split", default=None,
                        choices=[None, "valid", "test"],
                        help="Merge another split into training. Validates on "
                             "'test' when merging 'valid' to avoid leakage.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.feat_dim is None:
        _FEAT_DIMS = {"pca512": 512, "resnet50": 2048, "baidu": 8576}
        args.feat_dim = _FEAT_DIMS[args.feature_type]

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.feature_type} features from {args.data_dir}...")
    train_matches, _, _ = load_matches(args.data_dir, "train", args.feature_type)
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

    # Reuse TSM's dataset so window sizing / label alignment is identical.
    # Framerate must match feature extraction rate (Baidu=1fps, PCA=2fps).
    from src.training.train_tsm import FEATURE_FRAMERATE as _FR
    _framerate = _FR.get(args.feature_type, 2)
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    model = NetVLADSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=args.hidden_dim, num_clusters=args.num_clusters,
        dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NetVLAD++ model: {n_params:,} params "
          f"(hidden={args.hidden_dim}, clusters={args.num_clusters})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=TSM_CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    weights = compute_class_weights_from_matches(
        train_matches, num_classes=TSM_CONFIG["num_classes"],
        bg_weight=TSM_CONFIG["bg_weight"],
    )
    print(f"Per-class weights: bg={weights[0]:.3f}, "
          f"events min={weights[1:].min():.2f} max={weights[1:].max():.2f}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))

    early_stop = EarlyStopping(patience=TSM_CONFIG["patience"], mode="min")

    import wandb
    wandb.init(project=args.wandb_project, name="netvlad",
               config={**TSM_CONFIG, **NETVLAD_DEFAULTS,
                       "feat_dim": args.feat_dim,
                       "feature_type": args.feature_type,
                       "extra_train_split": args.extra_train_split})

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 args.device)
        val_loss = validate_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        wandb.log({"train/loss": train_loss, "val/loss": val_loss,
                   "lr": scheduler.get_last_lr()[0], "epoch": epoch})
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, "best.pt"))

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # Generate and evaluate predictions on the validation split that was
    # used for early stopping (=test when extra_train_split=valid).
    print(f"Generating predictions on {val_split_name} split...")
    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_ResNET_TF2.npy", "2_ResNET_TF2.npy"),
        "baidu": ("1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"),
    }
    pred_dir = os.path.join(args.output_dir, "predictions")
    from src.training.train_tsm import FEATURE_SCALE as _FS
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=pred_dir, feature_files=feat_map[args.feature_type],
        window_size=TSM_CONFIG["window_size"], stride=TSM_CONFIG["stride"],
        nms_window=TSM_CONFIG["nms_window"],
        confidence_threshold=TSM_CONFIG["confidence_threshold"],
        framerate=_framerate, device=args.device,
        feature_scale=_FS.get(args.feature_type, 1.0),
    )

    print("Running SoccerNet evaluation...")
    results = run_evaluation(args.data_dir, pred_dir, split=val_split_name)
    avg_map = float(results.get("a_mAP", 0.0)) * 100.0
    print(f"\nmAP Results (NetVLAD++ single-model on {val_split_name}):")
    print(f"  avg-mAP tight: {avg_map:.4f}%")
    wandb.log({"eval/avg_mAP_tight": avg_map})
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
