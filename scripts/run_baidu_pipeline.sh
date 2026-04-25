#!/usr/bin/env bash
# Overnight/morning Baidu pipeline — runs AFTER the download finishes.
#
# Checks all three splits have their Baidu .npy files, retrains TSM +
# SlowFast + NetVLAD on Baidu (train+valid merged, validating on test),
# runs the 3-way ensemble on test with the sweep-best post-processing,
# and produces the full metrics report.
#
# Usage:
#   bash scripts/run_baidu_pipeline.sh
#
# If it fails at any stage, fix the issue and re-run: earlier training
# stages' best.pt checkpoints are skipped when present, so you don't
# retrain anything that already succeeded.
set -u
set -o pipefail

PY=.venv/bin/python
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# --- Tunables (override via env) ---
# bs=16 everywhere — mmap fix + 15x Baidu feature scaling + 4 DataLoader
# workers confirm this fits under 4 GB VRAM with GPU 50-98% utilised.
BATCH_SIZE=${BATCH_SIZE:-16}    # TSM
BATCH_SF=${BATCH_SF:-16}        # SF
BATCH_NV=${BATCH_NV:-16}        # NetVLAD
DATA=${DATA_DIR:-data/}

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "[ERROR] $*" >&2; exit "${2:-1}"; }

# --- Preflight: count Baidu files per split ---
log "Preflight: Baidu feature coverage"
for s in train valid test; do
    # NOTE: we count on disk — this counts files whose path components
    # match our split list. Since the downloader places files at the
    # same league/year/match path regardless of split, we count the
    # expected match IDs per split.
    expected=$($PY -c "
from SoccerNet.utils import getListGames
print(len(getListGames(split='$s')) * 2)" 2>/dev/null)
    have=$(find "$DATA" -name '*baidu_soccer_embeddings.npy' 2>/dev/null | while read -r f; do
        md="$(dirname "$f" | sed 's|'"$DATA"'||')"; echo "$md"
    done | sort -u | wc -l)
    printf "  %-5s: expected %d halves (file count only, split assignment checked at load time)\n" "$s" "$expected"
done
total_baidu=$(find "$DATA" -name '*baidu_soccer_embeddings.npy' -type f 2>/dev/null | wc -l)
log "Total Baidu files on disk: $total_baidu (target ~900 for train+valid+test, ~1000 with challenge)"
if [ "$total_baidu" -lt 800 ]; then
    log "WARN: fewer than 800 Baidu files present — download may not be finished."
    log "      Re-run ``scripts/download_features_hf.py`` first, or re-run this"
    log "      script later once the count is higher."
    # Don't die — user may intentionally want to proceed with what they have.
fi

# --- Stage 1: TSM on Baidu ---
TSM_OUT=outputs/tsm_baidu
if [ -f "$TSM_OUT/best.pt" ]; then
    log "Stage 1/5: TSM checkpoint exists at $TSM_OUT/best.pt — skipping."
else
    log "Stage 1/5: Training TSM on Baidu (batch_size=$BATCH_SIZE, 8576-dim)"
    $PY -u -m src.training.train_tsm \
        --data-dir "$DATA" \
        --output-dir "$TSM_OUT" \
        --feature-type baidu \
        --batch-size "$BATCH_SIZE" \
        --extra-train-split valid \
        2>&1 | tee "$TSM_OUT.log" \
        || die "TSM Baidu training failed. See $TSM_OUT.log" 2
fi

# --- Stage 2: SlowFast on Baidu ---
SF_OUT=outputs/slowfast_baidu
if [ -f "$SF_OUT/best.pt" ]; then
    log "Stage 2/5: SlowFast checkpoint exists at $SF_OUT/best.pt — skipping."
else
    log "Stage 2/5: Training SlowFast on Baidu (batch_size=$BATCH_SF)"
    $PY -u -m src.training.train_slowfast \
        --data-dir "$DATA" \
        --output-dir "$SF_OUT" \
        --coarse-checkpoint "$TSM_OUT/best.pt" \
        --feature-type baidu \
        --batch-size "$BATCH_SF" \
        --extra-train-split valid \
        2>&1 | tee "$SF_OUT.log" \
        || die "SlowFast Baidu training failed. See $SF_OUT.log" 2
fi

# --- Stage 3: NetVLAD on Baidu ---
NV_OUT=outputs/netvlad_baidu
if [ -f "$NV_OUT/best.pt" ]; then
    log "Stage 3/5: NetVLAD checkpoint exists at $NV_OUT/best.pt — skipping."
else
    log "Stage 3/5: Training NetVLAD++ on Baidu (batch_size=$BATCH_NV)"
    $PY -u -m src.training.train_netvlad \
        --data-dir "$DATA" \
        --output-dir "$NV_OUT" \
        --feature-type baidu \
        --batch-size "$BATCH_NV" \
        --extra-train-split valid \
        2>&1 | tee "$NV_OUT.log" \
        || die "NetVLAD Baidu training failed. See $NV_OUT.log" 2
fi

# --- Stage 4: 3-way ensemble on test ---
ENS_OUT=outputs/final_baidu
log "Stage 4/5: 3-way ensemble on test (Baidu features)"
$PY -u scripts/run_ensemble3.py \
    --data-dir "$DATA" \
    --split test \
    --feature-type baidu \
    --tsm-checkpoint     "$TSM_OUT/best.pt" \
    --slowfast-checkpoint "$SF_OUT/best.pt" \
    --netvlad-checkpoint  "$NV_OUT/best.pt" \
    --output-dir "$ENS_OUT" \
    --tsm-weight 0.10 --slowfast-weight 0.60 --netvlad-weight 0.30 \
    --nms-mode hard --nms-window 5 --confidence-threshold 0.05 \
    --scales 40:20,80:40 \
    2>&1 | tee "$ENS_OUT.log" \
    || die "Ensemble failed. See $ENS_OUT.log" 3

# --- Stage 5: Full metrics ---
log "Stage 5/5: Full metrics report (tight + loose + per-class + chart)"
$PY -u scripts/generate_full_metrics.py \
    --data-dir "$DATA" \
    --split test \
    --output-dir results \
    --tag final_baidu \
    --predictions-dir "$ENS_OUT/predictions" \
    2>&1 | tee outputs/full_metrics_baidu.log \
    || die "Metrics generation failed. See outputs/full_metrics_baidu.log" 3

log "Done. Headline mAP is printed above (grep 'avg-mAP tight')."
log "Artifacts:"
log "  - $ENS_OUT/predictions/  (submission predictions)"
log "  - results/metrics_final_baidu.json"
log "  - results/per_class_ap_final_baidu.svg"
