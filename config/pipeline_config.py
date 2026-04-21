"""Two-stage pipeline configuration."""

PIPELINE_CONFIG = {
    # Must match the hidden_dim of the trained TSM coarse checkpoint
    # (TSM_CONFIG["hidden_dim"]=256). Mismatching here makes
    # load_checkpoint() in train_slowfast.py raise a state_dict shape error
    # at the two-stage evaluation step.
    "coarse_hidden_dim": 256,
    "coarse_confidence_threshold": 0.15, # low threshold = high recall
    "coarse_nms_window": 20,            # 10s at 2fps
    "coarse_window_size": 40,
    "coarse_stride": 20,
    "fine_pad_frames": 20,              # pad candidates to 80 frames
    "fine_nms_window": 10,              # 5s at 2fps, tighter for fine stage
    "fine_confidence_threshold": 0.2,
    "framerate": 2,
}
