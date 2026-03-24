"""SlowFast model and training hyperparameters."""

SLOWFAST_CONFIG = {
    "feat_dim": 512,
    "num_classes": 17,
    "slow_stride": 4,
    "hidden_dim": 256,
    "batch_size": 8,
    "chunk_size": 80,       # 40s at 2fps, larger window for slow pathway
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 40,
    "patience": 10,
    "bg_weight": 0.05,
}
