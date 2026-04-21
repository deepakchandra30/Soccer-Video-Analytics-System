"""TSM model and training hyperparameters."""

TSM_CONFIG = {
    "feat_dim": 512,
    "num_classes": 17,
    "hidden_dim": 256,
    "n_shifts": 2,
    "n_div": 8,
    "chunk_size": 40,       # frames (20s at 2fps)
    "event_ratio": 0.7,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,
    "bg_weight": 0.05,
    # 15s NMS was too wide for tight-mAP: the evaluator averages tolerance
    # from 1-5s, so any two same-class events within 15s would have one
    # suppressed. 15 frames = 7.5s balances recall against duplicate spikes.
    "nms_window": 15,       # frames (7.5s at 2fps)
    "confidence_threshold": 0.2,
    "framerate": 2,
    "window_size": 40,      # inference sliding window
    "stride": 20,           # inference stride
}
