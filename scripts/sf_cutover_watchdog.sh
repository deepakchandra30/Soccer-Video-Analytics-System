#!/usr/bin/env bash
# Watchdog #3: cap SlowFast-Baidu training at CUTOVER_HHMM so NetVLAD-Baidu
# has a training window before the NV watchdog's 11:30 hard cap. Same
# mechanism as TSM watchdog: kill SF python, kill pipeline, relaunch —
# pipeline will skip stages 1 (TSM best.pt exists) and 2 (SF best.pt exists)
# and start NetVLAD at bs=16.
set -u
CUTOVER_HHMM=${CUTOVER_HHMM:-11:05}
cutover_ts=$(date -d "today $CUTOVER_HHMM" +%s)
echo "[sf-watchdog] cutover at $CUTOVER_HHMM IST (unix=$cutover_ts)"
LOG=outputs/slowfast_baidu.log

while true; do
    now=$(date +%s)
    # If pipeline already advanced past SF, exit.
    if grep -qE "Stage 3/5|NetVLAD checkpoint exists" outputs/baidu_pipeline.log 2>/dev/null; then
        echo "[sf-watchdog $(date +%H:%M:%S)] SF already done — stepping aside."
        exit 0
    fi
    # Natural early stop?
    if grep -qE "Early stopping" "$LOG" 2>/dev/null; then
        echo "[sf-watchdog $(date +%H:%M:%S)] SF early-stopped naturally — stepping aside."
        exit 0
    fi
    if [ "$now" -ge "$cutover_ts" ]; then
        echo "[sf-watchdog $(date +%H:%M:%S)] HARD CUTOVER — killing SF so NetVLAD has runway."
        break
    fi
    sleep 20
done

# Kill SF training + current pipeline shell so we can relaunch cleanly.
pkill -f "python.*train_slowfast" 2>/dev/null
sleep 3
pkill -KILL -f "python.*train_slowfast" 2>/dev/null
pkill -f "run_baidu_pipeline.sh" 2>/dev/null
sleep 2

if [ ! -f "outputs/slowfast_baidu/best.pt" ]; then
    echo "[sf-watchdog] ERROR: no SlowFast best.pt — cannot proceed. Bailing."
    exit 1
fi
echo "[sf-watchdog] SF checkpoint present ($(stat -c %s outputs/slowfast_baidu/best.pt) bytes). Relaunching pipeline from stage 3."

mv outputs/baidu_pipeline.log outputs/baidu_pipeline_sf_phase.log 2>/dev/null
nohup bash scripts/run_baidu_pipeline.sh > outputs/baidu_pipeline.log 2>&1 &
echo "[sf-watchdog $(date +%H:%M:%S)] pipeline relaunched PID=$!"
