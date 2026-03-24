"""Ablation experiment configurations for Phase 2 evaluation."""

ABLATION_EXPERIMENTS = [
    # backbone feature dimension ablation (TAR-03)
    {"name": "tsm_pca512", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 40,
     "pipeline": "single"},
    {"name": "tsm_resnet50_2048", "model": "tsm", "feat_dim": 2048,
     "feature_type": "resnet50", "hidden_dim": 256, "chunk_size": 40,
     "pipeline": "single"},

    # hidden dimension ablation (TAR-06)
    {"name": "tsm_hidden128", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 128, "chunk_size": 40,
     "pipeline": "single"},
    {"name": "tsm_hidden512", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 512, "chunk_size": 40,
     "pipeline": "single"},

    # temporal window size ablation (TAR-06)
    {"name": "tsm_window20", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 20,
     "pipeline": "single"},
    {"name": "tsm_window40", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 40,
     "pipeline": "single"},
    {"name": "tsm_window60", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 60,
     "pipeline": "single"},
    {"name": "tsm_window120", "model": "tsm", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 120,
     "pipeline": "single"},

    # single-stage vs two-stage (TAR-06, TSP)
    {"name": "slowfast_single", "model": "slowfast", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 80,
     "pipeline": "single"},
    {"name": "two_stage_pca512", "model": "two_stage", "feat_dim": 512,
     "feature_type": "pca512", "hidden_dim": 256, "chunk_size": 40,
     "pipeline": "two_stage", "coarse_hidden_dim": 128, "fine_chunk_size": 80},
    {"name": "two_stage_resnet50", "model": "two_stage", "feat_dim": 2048,
     "feature_type": "resnet50", "hidden_dim": 256, "chunk_size": 40,
     "pipeline": "two_stage", "coarse_hidden_dim": 128, "fine_chunk_size": 80},
]
