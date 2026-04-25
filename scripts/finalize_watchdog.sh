#!/usr/bin/env bash
# Failsafe watchdog: at CUTOVER_HHMM (default 11:40), kill any remaining
# training and force the final ensemble+metrics with whatever Baidu
# checkpoints exist. Ensures something ships by 12:00 IST deadline.
set -u
CUTOVER_HHMM=${CUTOVER_HHMM:-11:40}
cutover_ts=$(date -d "today $CUTOVER_HHMM" +%s)
echo "[finalize] cutover at $CUTOVER_HHMM IST (unix=$cutover_ts)"

TSM=outputs/tsm_baidu/best.pt
SF=outputs/slowfast_baidu/best.pt
NV=outputs/netvlad_baidu/best.pt

while true; do
    now=$(date +%s)
    # If metrics already produced, we're done.
    if [ -f "results/metrics_final_baidu.json" ] || [ -f "results/metrics_final_baidu_2way.json" ]; then
        echo "[finalize $(date +%H:%M:%S)] metrics already produced — stepping aside."
        exit 0
    fi
    if [ "$now" -ge "$cutover_ts" ]; then
        echo "[finalize $(date +%H:%M:%S)] HARD CUTOVER — forcing ensemble now."
        break
    fi
    sleep 20
done

# Kill any training still running + any pipeline shell.
pkill -f "python.*train_netvlad" 2>/dev/null
pkill -f "python.*train_slowfast" 2>/dev/null
pkill -f "python.*train_tsm" 2>/dev/null
sleep 3
pkill -KILL -f "python.*train_" 2>/dev/null
pkill -f "run_baidu_pipeline.sh" 2>/dev/null
sleep 2

# Decide which ensemble based on what's trained.
if [ -f "$TSM" ] && [ -f "$SF" ] && [ -f "$NV" ]; then
    MODE=3way
    ENS_OUT=outputs/final_baidu
    echo "[finalize] 3-way Baidu ensemble (TSM+SF+NV)"
    .venv/bin/python -u scripts/run_ensemble3.py \
        --data-dir data/ --split test --feature-type baidu \
        --tsm-checkpoint "$TSM" --slowfast-checkpoint "$SF" --netvlad-checkpoint "$NV" \
        --output-dir "$ENS_OUT" \
        --tsm-weight 0.10 --slowfast-weight 0.60 --netvlad-weight 0.30 \
        --scales 40:20,80:40 --nms-mode hard --nms-window 5 --confidence-threshold 0.05 \
        2>&1 | tee outputs/final_baidu_finalize.log
    TAG=final_baidu
elif [ -f "$TSM" ] && [ -f "$SF" ]; then
    MODE=2way
    ENS_OUT=outputs/final_baidu_2way
    echo "[finalize] 2-way Baidu ensemble (TSM+SF, no NetVLAD)"
    .venv/bin/python -u scripts/run_ensemble.py \
        --data-dir data/ --split test --feature-type baidu \
        --tsm-checkpoint "$TSM" --slowfast-checkpoint "$SF" \
        --output-dir "$ENS_OUT" \
        --tsm-weight 0.15 \
        --scales 40:20,80:40 --nms-mode hard --nms-window 5 --confidence-threshold 0.05 \
        2>&1 | tee outputs/final_baidu_2way.log
    TAG=final_baidu_2way
else
    echo "[finalize] ERROR: missing TSM or SF Baidu checkpoint. Cannot ensemble."
    exit 1
fi

echo "[finalize] Generating full metrics ($MODE)..."
.venv/bin/python -u scripts/generate_full_metrics.py \
    --data-dir data/ --split test --output-dir results --tag "$TAG" \
    --predictions-dir "$ENS_OUT/predictions" \
    2>&1 | tee "outputs/full_metrics_${TAG}.log"

echo "[finalize $(date +%H:%M:%S)] DONE mode=$MODE"
grep "avg-mAP" outputs/final_baidu_finalize.log outputs/final_baidu_2way.log 2>/dev/null | tail -2
