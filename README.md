# Soccer Video Analytics System

An end-to-end pipeline for soccer match analysis using temporal deep learning, computer vision tracking, and LLM-powered narrative generation. Built on the SoccerNet-v3 benchmark.

## Overview

This system transforms raw match footage into structured analytics through four stages:

1. **Temporal Action Recognition** - TSM and SlowFast models detect 17 event types (goals, fouls, cards, etc.) with per-frame scoring and NMS post-processing
2. **Player Tracking** - YOLOv8 detection + ByteTrack multi-object tracking with camera cut handling and pitch coordinate projection via homography
3. **Analytics & Narratives** - Event-player attribution, per-player heatmaps and screen-time stats, LLM-generated match reports with deterministic fact-checking
4. **Interactive Dashboard** - React web UI with synchronized video playback, event timeline, player analytics panels, and D3.js pitch visualizations

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
```

### Download Features (no NDA needed)

Three feature sets are available. The **headline result uses Baidu features**; use
the HuggingFace mirror because KAUST's official host has been unreachable since
2026-04-19.

```bash
# RECOMMENDED - Baidu 8576-dim features, via HuggingFace mirror.
# Headline 61.11% tight / 74.20% loose was produced on these features.
# Set HF_TOKEN (free at huggingface.co/settings/tokens) for 10-20x throughput.
export HF_TOKEN=<your_hf_read_token>
python scripts/download_features_hf.py --local-dir data/ --splits train valid test

# Alternative - PCA-512 (ResNet-152 projected to 512-dim, 2fps, ~3GB total).
# Faster to download, lower mAP ceiling (~42% tight). Good for smoke tests.
python scripts/download_features.py --local-dir data/ --features pca512

# Alternative - ResNet-152 raw 2048-dim (between the two above).
python scripts/download_features.py --local-dir data/ --features resnet50
```

### Download Videos (NDA required)

```bash
python scripts/download_features.py --local-dir data/ --video --password YOUR_NDA_PASSWORD
```

### Training

Training scripts log to Weights & Biases. If you don't have a W&B
account, prefix the commands with `WANDB_MODE=offline` to write run logs
locally without needing an API key. All scripts accept `--feature-type
{pca512,resnet50,baidu}`; pass `baidu` to train on the 8576-dim features
used for the headline.

**Clean protocol (default, honest reporting)** - train on `train`,
validate on `valid`, never touch `test` until the final ensemble eval.
Pass `--seed` to enable multi-seed reliability runs.

```bash
# TSM on Baidu features (clean: val=valid, no extra-train-split)
WANDB_MODE=offline python -m src.training.train_tsm \
  --data-dir data/ --output-dir outputs/tsm_baidu \
  --feature-type baidu --seed 42

# SlowFast (two-stage, needs TSM checkpoint as coarse)
WANDB_MODE=offline python -m src.training.train_slowfast \
  --data-dir data/ --output-dir outputs/slowfast_baidu \
  --coarse-checkpoint outputs/tsm_baidu/best.pt \
  --feature-type baidu --seed 42

# NetVLAD++ (third ensemble member)
WANDB_MODE=offline python -m src.training.train_netvlad \
  --data-dir data/ --output-dir outputs/netvlad_baidu \
  --feature-type baidu --seed 42

# run ablation studies
python scripts/run_ablation.py --data-dir data/ --output-dir outputs/ablation \
  --checkpoints-dir outputs/ --experiments all
```

**Legacy protocol (test-set leakage)** - only for reproducing the
previously-published 61.11% headline. Each command requires
`--allow-test-leakage` and prints a banner.

```bash
WANDB_MODE=offline python -m src.training.train_tsm \
  --data-dir data/ --output-dir outputs/tsm_baidu_legacy \
  --feature-type baidu --extra-train-split valid --allow-test-leakage
```

### Reproducing the Headline mAP

The project ships **two evaluation protocols** so the bias from test-set
tuning is explicit (the supervisor's 2026-04-30 review flagged the earlier
protocol):

- **Clean held-out (default for honest reporting)** - train on `train`,
  tune on `valid`, test once on `test`, multi-seed with mean ± std and
  bootstrap 95% CI. Run with
  `python scripts/run_multi_seed.py --feature-type baidu` (default 3
  seeds, ~3-4× legacy compute). Aggregated numbers in
  `results/clean_protocol/multi_seed_summary.{json,md}`.
- **Legacy test-tuned (kept only for reproducing the old number)** -
  train on `train+valid` merged, early-stop / ensemble-select on `test`.
  This produced **61.11% tight / 74.20% loose**
  (`results/metrics_final_baidu.json`). It can be reproduced with
  `LEGACY=1 bash scripts/run_baidu_pipeline.sh`; the leakage banner
  prints at startup.

See `docs/EVALUATION_PROTOCOL.md` for the full breakdown of which knobs
were tuned on test, why each was a leak, and how the clean pipeline
neutralizes them. The side-by-side comparison is regenerated with
`python scripts/generate_protocol_comparison.py` and lives at
`results/PROTOCOL_COMPARISON.md`.

**Fast path - download our pretrained checkpoints (recommended, ≈3 min):**

```bash
python scripts/download_pretrained.py    # 464 MB from a private GitHub Release
LEGACY=1 bash scripts/run_baidu_pipeline.sh   # legacy headline (all train stages skip)
# OR for the clean held-out number:
bash scripts/run_clean_pipeline.sh            # one clean seed; multi-seed via run_multi_seed.py
```

This needs `gh` CLI authenticated against the repo (it already is if you
cloned via `gh repo clone` or `git clone` with a credential helper). The
download is sha256-verified against `results/checkpoints_manifest.json` so
silent corruption can't occur.

**Slow path - train from scratch (4-12 h on a single 24 GB GPU):**

```bash
# Honest reporting: clean held-out with N seeds (mean ± std + 95% CI):
python scripts/run_multi_seed.py --feature-type baidu

# Legacy headline reproduction (test-set leakage, banner-gated):
LEGACY=1 bash scripts/run_baidu_pipeline.sh
```

This produces `results/metrics_final_baidu.json` and the submission
predictions in `outputs/final_baidu/predictions/`. Each training stage is
skipped if its `best.pt` already exists, so re-runs after tuning ensemble
knobs do not retrain from scratch. Note: training is non-deterministic on
GPU (CUDA + DataLoader workers), so a freshly-trained run may land in the
54-58% range rather than exactly 61.11% - the fast path above bypasses
this variance entirely.

For the earlier PCA-512 result (42.37% tight, TSM+SlowFast only, faster to
reproduce), use the 2-stage orchestrator:

```bash
python scripts/run_full_pipeline.py --data-dir data/ --feature-type pca512
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

### 3-Way Ensemble (Headline Configuration)

The headline 61.11% tight mAP comes from fusing TSM, SlowFast, and NetVLAD++
predictions on Baidu 8576-dim features, weighted 0.10 / 0.60 / 0.30 with
hard-NMS (window 5) and multi-scale (40:20, 80:40) post-processing. See
`scripts/run_ensemble3.py` for the fusion logic and `scripts/run_baidu_pipeline.sh`
for the exact flags.

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
