#!/usr/bin/env bash
# Watchdog #2: ensure some submission ships by 12:00 IST even if NetVLAD
# Baidu training overruns. At CUTOVER_HHMM (default 11:30) check the state.
# If netvlad_baidu/best.pt is missing, kill the pipeline and run the 2-way
# Baidu ensemble (TSM + SlowFast) so the submission is still Baidu-based.
set -u
CUTOVER_HHMM=${CUTOVER_HHMM:-11:30}
cutover_ts=$(date -d "today $CUTOVER_HHMM" +%s)
echo "[nv-watchdog] cutover at $CUTOVER_HHMM IST (unix=$cutover_ts)"

TSM=outputs/tsm_baidu/best.pt
SF=outputs/slowfast_baidu/best.pt
NV=outputs/netvlad_baidu/best.pt

while true; do
    now=$(date +%s)
    if [ -f "$NV" ]; then
        echo "[nv-watchdog $(date +%H:%M:%S)] NetVLAD checkpoint exists — stepping aside, pipeline will 3-way ensemble."
        exit 0
    fi
    # If the ensemble stage started (3-way), we're good — exit.
    if grep -qE "Stage 4/5|avg-mAP tight" outputs/baidu_pipeline.log 2>/dev/null; then
        echo "[nv-watchdog $(date +%H:%M:%S)] ensemble stage detected — stepping aside."
        exit 0
    fi
    if [ "$now" -ge "$cutover_ts" ]; then
        echo "[nv-watchdog $(date +%H:%M:%S)] HARD CUTOVER — NetVLAD not done, forcing 2-way Baidu ensemble."
        break
    fi
    sleep 20
done

# Kill anything still training
pkill -f "python.*train_netvlad" 2>/dev/null
sleep 3
pkill -KILL -f "python.*train_netvlad" 2>/dev/null
pkill -f "run_baidu_pipeline.sh" 2>/dev/null
sleep 2

if [ ! -f "$TSM" ] || [ ! -f "$SF" ]; then
    echo "[nv-watchdog] ERROR: missing TSM or SF Baidu checkpoint. Cannot run 2-way."
    echo "[nv-watchdog] TSM: $TSM exists=$([ -f "$TSM" ] && echo yes || echo no)"
    echo "[nv-watchdog] SF:  $SF exists=$([ -f "$SF" ] && echo yes || echo no)"
    exit 1
fi

ENS_OUT=outputs/final_baidu_2way
echo "[nv-watchdog $(date +%H:%M:%S)] Running 2-way Baidu ensemble (TSM + SF, weights 0.15/0.85)..."
.venv/bin/python -u scripts/run_ensemble.py \
    --data-dir data/ \
    --split test \
    --feature-type baidu \
    --tsm-checkpoint "$TSM" \
    --slowfast-checkpoint "$SF" \
    --output-dir "$ENS_OUT" \
    --tsm-weight 0.15 \
    --scales 40:20,80:40 \
    --nms-mode hard --nms-window 5 --confidence-threshold 0.05 \
    2>&1 | tee outputs/baidu_2way_ensemble.log

echo "[nv-watchdog $(date +%H:%M:%S)] Generating full metrics..."
.venv/bin/python -u scripts/generate_full_metrics.py \
    --data-dir data/ \
    --split test \
    --output-dir results \
    --tag final_baidu_2way \
    --predictions-dir "$ENS_OUT/predictions" \
    2>&1 | tee outputs/full_metrics_baidu_2way.log

echo "[nv-watchdog $(date +%H:%M:%S)] DONE. Headline mAP:"
grep "avg-mAP tight" outputs/baidu_2way_ensemble.log | tail -1
