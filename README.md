# Soccer Video Analytics System

An end-to-end pipeline for soccer match analysis using temporal deep learning, computer vision tracking, and LLM-powered narrative generation. Built on the SoccerNet-v3 benchmark.

## Overview

This system transforms raw match footage into structured analytics through four stages:

1. **Temporal Action Recognition** — TSM and SlowFast models detect 17 event types (goals, fouls, cards, etc.) with per-frame scoring and NMS post-processing
2. **Player Tracking** — YOLOv8 detection + ByteTrack multi-object tracking with camera cut handling and pitch coordinate projection via homography
3. **Analytics & Narratives** — Event-player attribution, per-player heatmaps and screen-time stats, LLM-generated match reports with deterministic fact-checking
4. **Interactive Dashboard** — React web UI with synchronized video playback, event timeline, player analytics panels, and D3.js pitch visualizations

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+ (for dashboard)
- CUDA-capable GPU (recommended for training/inference)
- SoccerNet NDA approval for video access

### Setup

```bash
# clone and install
git clone https://github.com/10srav/Soccer-Video-Analytics-System-.git
cd Soccer-Video-Analytics-System-
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt

# download SoccerNet features (no NDA needed)
python scripts/download_features.py --data-dir data/

# download videos (NDA required)
python scripts/download_features.py --data-dir data/ --video --password YOUR_NDA_PASSWORD
```

### Training

Both training scripts log to Weights & Biases. If you don't have a W&B
account, prefix the commands with `WANDB_MODE=offline` to write run logs
locally without needing an API key:

```bash
# train TSM baseline
WANDB_MODE=offline python -m src.training.train_tsm --data-dir data/ --output-dir outputs/tsm_baseline

# train SlowFast + two-stage evaluation
WANDB_MODE=offline python -m src.training.train_slowfast --data-dir data/ --output-dir outputs/slowfast \
  --coarse-checkpoint outputs/tsm_baseline/best.pt

# run ablation studies
python scripts/run_ablation.py --data-dir data/ --output-dir outputs/ablation \
  --checkpoints-dir outputs/ --experiments all
```

### Tracking

```bash
python scripts/run_tracking.py --video data/path/to/match/1_720p.mkv \
  --output-dir outputs/tracking
```

### Analytics & Narratives

```bash
python scripts/run_analytics.py \
  --events-json outputs/tsm_baseline/predictions/match_id/results_spotting.json \
  --tracks-json outputs/tracking/tracks.json \
  --output-dir outputs/analytics \
  --generate-narrative
```

### Dashboard

```bash
# start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# start React dashboard (in a separate terminal)
cd dashboard
npm install
npm start
```

The dashboard will be available at `http://localhost:3000` and connects to the API at `http://localhost:8000`.

## Project Structure

```
src/
  data/           # SoccerNet dataset loaders and utilities
  models/         # TSM and SlowFast temporal models
    temporal/     # TemporalShift, TSMSpottingHead, SlowFastSpotting
  training/       # Training loops and training scripts
  evaluation/     # SoccerNet evaluation, ablation, benchmarking
  tracking/       # YOLOv8 detection, ByteTrack, homography
  analytics/      # Event-player attribution, match statistics
  narratives/     # LLM generation, fact-checking, BLEU evaluation
  api/            # FastAPI REST endpoints
config/           # Hyperparameter configs (TSM, SlowFast, pipeline, ablation)
scripts/          # CLI entry points
dashboard/        # React frontend
tests/            # Comprehensive test suite (134+ tests)
docs/literature/  # Literature review and SOTA comparison
```

## Architecture

### Two-Stage Pipeline (Novel Contribution)

The core research contribution is a two-stage inference pipeline:

- **Stage 1 (TSM)**: Lightweight temporal model scans full matches at high speed, generating candidate event windows with high recall (~90%) at a low confidence threshold
- **Stage 2 (SlowFast)**: Dual-pathway model re-classifies only the candidate windows at higher temporal resolution, improving precision while processing only 5-15% of total frames

This achieves 2-5x inference speedup over single-stage approaches while maintaining competitive mAP on SoccerNet-v3.

### Evaluation Protocol

All results use the official SoccerNet evaluation SDK with `version=2` and `metric="tight"` (avg-mAP over 1-5 second tolerance windows), ensuring direct comparability with published baselines.

## Testing

```bash
# run full test suite
python -m pytest tests/ -v

# run specific phase tests
python -m pytest tests/phase2/ -v  # temporal models
python -m pytest tests/phase3/ -v  # tracking
python -m pytest tests/phase4/ -v  # analytics/narratives
python -m pytest tests/phase5/ -v  # API
```

## Dependencies

Key packages:
- PyTorch 2.2.0 + torchvision 0.17.0 (CUDA 12.1)
- SoccerNet 0.1.62
- ultralytics 8.4.22 (YOLOv8)
- supervision 0.27.0 (ByteTrack)
- FastAPI 0.135.1
- React 18 + TypeScript + Zustand + D3.js

See `requirements.txt` for the complete pinned dependency list.

## License

This project uses the SoccerNet dataset under its original license terms. See the SoccerNet NDA for data usage conditions.
