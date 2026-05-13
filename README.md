# Soccer Video Analytics System

End-to-end pipeline for soccer match analysis using temporal deep learning, computer vision tracking, and LLM-assisted narrative generation.

## Overview

This system transforms match footage into structured analytics through four stages:

1. **Temporal action spotting** - TSM and SlowFast detect 17 event types with per-timestep scoring and NMS post-processing.
2. **Player tracking** - YOLOv8 detection + ByteTrack multi-object tracking with camera-cut handling and pitch coordinate projection via homography.
3. **Analytics and narratives** - event-player attribution, per-player heatmaps and screen-time stats, and match reports with deterministic fact checks.
4. **Interactive dashboard** - React UI with synchronized video playback, event timeline, analytics panels, and D3 pitch visualizations.

## Quick start

### Prerequisites

- Python 3.12+
- Node.js 18+ (dashboard)
- CUDA-capable GPU (recommended)
- SoccerNet NDA approval for videos

### Setup

```bash
git clone https://github.com/10srav/Soccer-Video-Analytics-System-.git
cd Soccer-Video-Analytics-System-
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Download features (no NDA needed)

Three feature sets are supported. The headline results use Baidu features.

```bash
# RECOMMENDED - Baidu 8576-dim features from the official SoccerNet (KAUST) host
python scripts/download_features.py --local-dir data/ --features baidu --splits train valid test

# Alternative - PCA-512 (projected features; smaller, lower ceiling)
python scripts/download_features.py --local-dir data/ --features pca512 --splits train valid test

# Alternative - ResNet-152 raw 2048-dim
python scripts/download_features.py --local-dir data/ --features resnet152 --splits train valid test
```

### Download videos (NDA required)

```bash
python scripts/download_features.py --local-dir data/ --video --password YOUR_NDA_PASSWORD
```

## Training

If you do not use Weights and Biases, set `WANDB_MODE=offline`.

All training scripts accept `--feature-type {pca512,resnet152,baidu}`. Use `baidu` to train on the 8576-dim features used for the headline.

```bash
# TSM on Baidu features
WANDB_MODE=offline python -m src.training.train_tsm \
  --data-dir data/ --output-dir outputs/tsm_baidu \
  --feature-type baidu --seed 42

# SlowFast (two-stage; uses TSM checkpoint)
WANDB_MODE=offline python -m src.training.train_slowfast \
  --data-dir data/ --output-dir outputs/slowfast_baidu \
  --coarse-checkpoint outputs/tsm_baidu/best.pt \
  --feature-type baidu --seed 42

# NetVLAD++ (third ensemble member)
WANDB_MODE=offline python -m src.training.train_netvlad \
  --data-dir data/ --output-dir outputs/netvlad_baidu \
  --feature-type baidu --seed 42
```

## Reproducing the headline mAP

This repo supports two evaluation protocols so the effect of test-set tuning is explicit.

### Protocol A - Clean held-out (recommended)

- Train on `train`
- Tune hyperparameters on `valid`
- Evaluate once on `test`
- Optional - run multiple seeds and report mean +- std (and CI if enabled)

Run:

```bash
python scripts/run_multi_seed.py --feature-type baidu
```

Outputs:
- `results/clean_protocol/multi_seed_summary.json`
- `results/clean_protocol/multi_seed_summary.md`

### Protocol B - Legacy test-tuned (for reproducing the previously reported headline)

This protocol is kept only to reproduce the earlier headline number:
- 61.11% tight / 74.20% loose (Baidu features)

It merges `train+valid` and uses `test` for early stopping / ensemble selection (this is test-set leakage and should not be used for new results).

Run:

```bash
LEGACY=1 bash scripts/run_baidu_pipeline.sh
```

Output:
- `results/metrics_final_baidu.json`

### Fast path - Use pretrained checkpoints (recommended)

```bash
python scripts/download_pretrained.py
LEGACY=1 bash scripts/run_baidu_pipeline.sh      # reproduces the legacy headline without retraining
# or for clean evaluation:
bash scripts/run_clean_pipeline.sh               # one clean seed (multi-seed via run_multi_seed.py)
```

Notes:
- Requires `gh` CLI authenticated for this repo (or any GitHub credential helper that can access releases).
- Checkpoints are SHA256-verified against `results/checkpoints_manifest.json`.

More details:
- `docs/EVALUATION_PROTOCOL.md`
- `results/PROTOCOL_COMPARISON.md` (regenerate with `python scripts/generate_protocol_comparison.py`)

## Tracking

```bash
python scripts/run_tracking.py --video data/path/to/match/1_720p.mkv \
  --output-dir outputs/tracking
```

## Analytics and narratives

```bash
python scripts/run_analytics.py \
  --events-json outputs/tsm_baseline/predictions/match_id/results_spotting.json \
  --tracks-json outputs/tracking/tracks.json \
  --output-dir outputs/analytics \
  --generate-narrative
```

## Dashboard

```bash
# API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# React dashboard
cd dashboard
npm install
npm start
```

Dashboard: `http://localhost:3000` (API: `http://localhost:8000`).

## Project structure

```
src/
  data/           # SoccerNet dataset loaders and utilities
  models/         # Temporal models
    temporal/     # TSM / SlowFast modules
  training/       # Training loops and scripts
  evaluation/     # Evaluation, ablations, benchmarking
  tracking/       # YOLOv8 + ByteTrack + homography
  analytics/      # Event-player attribution, match statistics
  narratives/     # Report generation and fact-checking
  api/            # FastAPI endpoints
config/           # Hyperparameter configs
scripts/          # CLI entry points
dashboard/        # React frontend
tests/            # Test suite
docs/             # Documentation and literature notes
```

## License

This project uses the SoccerNet dataset under its original license terms. See the SoccerNet NDA for data usage conditions.
