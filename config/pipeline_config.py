"""Two-stage pipeline configuration.

NOTE on framerate: it is *not* a key here on purpose. PCA-512 / ResNet-50 ship
at 2fps but Baidu ships at 1fps, so framerate is a per-feature-type runtime
value, not a pipeline hyperparameter. Pass it as the required ``framerate=``
keyword to ``TwoStagePipeline(...)`` and ``benchmark_pipeline(...)``. The
old static default of 2 silently halved every Baidu prediction position and
collapsed tight-mAP to ~1% — it is now mandatory at the call-site so the
mistake cannot recur.
"""

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
}
