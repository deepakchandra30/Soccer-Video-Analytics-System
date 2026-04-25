#!/usr/bin/env bash
# Watchdog around scripts/download_features_hf.py.
#
# Problem: huggingface_hub sometimes deadlocks with hf_transfer + many
# workers, leaving all threads in futex_do_wait. The process stays alive
# but stops making progress. Killing and restarting recovers instantly
# because the HF cache + our resumable logic skip everything already
# downloaded.
#
# What this script does:
#   - Launches the downloader as a child process.
#   - Every ~3 minutes, counts how many Baidu .npy files exist on disk.
#   - If the count has not increased in two consecutive checks (6 min
#     of zero progress), kills the child and restarts it.
#   - Exits cleanly when the target file count is reached.
#
# Usage:
#   nohup bash scripts/baidu_watchdog.sh > outputs/baidu_watchdog.log 2>&1 &
#
# Stop with:
#   pkill -f baidu_watchdog.sh
set -u

TARGET_COUNT=${TARGET_COUNT:-900}   # train(600) + valid(200) + test(200) = 1000; 900 is a safe "done enough" threshold
CHECK_INTERVAL=${CHECK_INTERVAL:-180}   # seconds between progress checks
STALL_CHECKS=${STALL_CHECKS:-2}   # number of consecutive zero-progress checks that triggers a restart

count_files() {
    find data -name '*baidu_soccer_embeddings*' -type f 2>/dev/null | wc -l
}

start_downloader() {
    echo "[$(date +%H:%M:%S)] starting downloader..."
    HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
        .venv/bin/python -u scripts/download_features_hf.py \
        --splits train valid test --local-dir data/ --workers 16 \
        > outputs/baidu_download.log 2>&1 &
    DL_PID=$!
    echo "[$(date +%H:%M:%S)] downloader PID=$DL_PID"
}

kill_downloader() {
    if [ -n "${DL_PID:-}" ] && kill -0 "$DL_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] killing downloader PID=$DL_PID"
        kill -9 "$DL_PID" 2>/dev/null
    fi
    pkill -9 -f download_features_hf.py 2>/dev/null || true
    sleep 2
}

trap 'echo "[$(date +%H:%M:%S)] watchdog received signal, killing downloader"; kill_downloader; exit 0' SIGTERM SIGINT

start_downloader
prev_count=$(count_files)
stall_count=0

while true; do
    # Exit condition: reached target
    cur=$(count_files)
    if [ "$cur" -ge "$TARGET_COUNT" ]; then
        echo "[$(date +%H:%M:%S)] target reached: $cur Baidu files on disk. Stopping."
        kill_downloader
        break
    fi

    sleep "$CHECK_INTERVAL"

    new=$(count_files)
    delta=$(( new - prev_count ))
    alive=0
    if [ -n "${DL_PID:-}" ] && kill -0 "$DL_PID" 2>/dev/null; then
        alive=1
    fi

    if [ "$delta" -eq 0 ]; then
        stall_count=$(( stall_count + 1 ))
        echo "[$(date +%H:%M:%S)] files=$new (no change), stall_count=$stall_count/$STALL_CHECKS, alive=$alive"
        if [ "$stall_count" -ge "$STALL_CHECKS" ] || [ "$alive" -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] RESTARTING downloader (stalled or died)"
            kill_downloader
            sleep 5
            start_downloader
            stall_count=0
        fi
    else
        rate=$(echo "scale=2; $delta / ($CHECK_INTERVAL / 60)" | bc 2>/dev/null || echo "?")
        echo "[$(date +%H:%M:%S)] files=$new (+$delta in ${CHECK_INTERVAL}s = $rate/min), alive=$alive"
        stall_count=0
    fi

    prev_count=$new
done

echo "[$(date +%H:%M:%S)] watchdog finished."
