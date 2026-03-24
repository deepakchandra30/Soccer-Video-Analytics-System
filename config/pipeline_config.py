"""Two-stage pipeline configuration."""

PIPELINE_CONFIG = {
    "coarse_hidden_dim": 128,           # lighter TSM for speed
    "coarse_confidence_threshold": 0.15, # low threshold = high recall
    "coarse_nms_window": 20,            # 10s at 2fps
    "coarse_window_size": 40,
    "coarse_stride": 20,
    "fine_pad_frames": 20,              # pad candidates to 80 frames
    "fine_nms_window": 10,              # 5s at 2fps, tighter for fine stage
    "fine_confidence_threshold": 0.2,
    "framerate": 2,
}
