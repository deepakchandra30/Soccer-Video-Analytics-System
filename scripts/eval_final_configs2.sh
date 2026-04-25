#!/usr/bin/env bash
# Second eval pass — narrow in on the weight region around config #6
# which hit 45.32% tight mAP (tsm_fixed + sf_fixed + netvlad_merged,
# w=0.15/0.55/0.30).
set -u
PY=.venv/bin/python
DATA=data/
SPLIT=test
FEATS=pca512
POST="--nms-mode hard --nms-window 5 --confidence-threshold 0.05 --scales 40:20,80:40"

log_line() { printf '\n==== %s ====\n' "$1"; }
run3() {
    # args: tag w1 w2 w3
    local tag=$1 w1=$2 w2=$3 w3=$4
    log_line "3-way mixed $tag  w=$w1/$w2/$w3"
    $PY scripts/run_ensemble3.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
      --tsm-checkpoint     outputs/tsm_fixed/best.pt \
      --slowfast-checkpoint outputs/slowfast_fixed/best.pt \
      --netvlad-checkpoint  outputs/netvlad_merged/best.pt \
      --output-dir outputs/final/$tag \
      --tsm-weight $w1 --slowfast-weight $w2 --netvlad-weight $w3 \
      $POST 2>&1 | grep "avg-mAP tight"
}

run3 mixed_A 0.20 0.55 0.25
run3 mixed_B 0.15 0.60 0.25
run3 mixed_C 0.10 0.60 0.30
run3 mixed_D 0.20 0.50 0.30
run3 mixed_E 0.15 0.50 0.35

echo ""
echo "Done."
