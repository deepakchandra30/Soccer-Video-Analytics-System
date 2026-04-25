#!/usr/bin/env bash
# Watchdog: wait for TSM-Baidu to early-stop OR 10:30 IST (hard cap),
# then kill the running pipeline cleanly and relaunch from SlowFast
# (which skips TSM because best.pt exists and uses bs=16).
set -u
LOG=outputs/tsm_baidu.log
CUTOVER_HHMM=${CUTOVER_HHMM:-10:30}

cutover_ts=$(date -d "today $CUTOVER_HHMM" +%s)
echo "[watchdog] cutover at $CUTOVER_HHMM IST (unix=$cutover_ts)"

while true; do
    now=$(date +%s)
    if [ "$now" -ge "$cutover_ts" ]; then
        echo "[watchdog $(date +%H:%M:%S)] hard cutover hit — killing TSM"
        reason=cutover
        break
    fi
    if grep -qE "Early stopping|Generating predictions|avg-mAP tight" "$LOG" 2>/dev/null; then
        echo "[watchdog $(date +%H:%M:%S)] TSM early-stopped naturally"
        reason=early_stop
        break
    fi
    if ! pgrep -f "python.*train_tsm" > /dev/null; then
        echo "[watchdog $(date +%H:%M:%S)] TSM process vanished — assuming done"
        reason=process_gone
        break
    fi
    sleep 20
done

# Kill anything still running so the pipeline shell exits cleanly.
pkill -f "python.*train_tsm" 2>/dev/null
sleep 3
pkill -KILL -f "python.*train_tsm" 2>/dev/null
pkill -f "run_baidu_pipeline.sh" 2>/dev/null
sleep 2

# Verify TSM checkpoint exists before moving on.
if [ ! -f "outputs/tsm_baidu/best.pt" ]; then
    echo "[watchdog] ERROR: no outputs/tsm_baidu/best.pt — cannot proceed. Bailing."
    exit 1
fi
echo "[watchdog] TSM checkpoint present ($(stat -c %s outputs/tsm_baidu/best.pt) bytes). Relaunching pipeline from stage 2."

# Archive current log, relaunch.
mv outputs/baidu_pipeline.log outputs/baidu_pipeline_tsm_phase.log 2>/dev/null
nohup bash scripts/run_baidu_pipeline.sh > outputs/baidu_pipeline.log 2>&1 &
echo "[watchdog $(date +%H:%M:%S)] pipeline relaunched PID=$! reason=$reason"
