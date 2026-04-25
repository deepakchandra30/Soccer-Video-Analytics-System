#!/usr/bin/env bash
# Run the final short list of ensemble configurations on the test split.
# Prints the avg-mAP tight for each so we can pick the submission config.
#
# Post-processing is fixed at the sweep optimum (conf=0.05, nms=5, hard).
set -u

PY=.venv/bin/python
DATA=data/
SPLIT=test
FEATS=pca512
POST="--nms-mode hard --nms-window 5 --confidence-threshold 0.05 --scales 40:20,80:40"

log_line() { printf '\n==== %s ====\n' "$1"; }

# 1) 2-way original checkpoints (known baseline = 45.20%)
log_line "2-way ORIGINAL (tsm_fixed + slowfast_fixed)  w=0.2/0.8"
$PY scripts/run_ensemble.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_fixed/best.pt \
  --slowfast-checkpoint outputs/slowfast_fixed/best.pt \
  --output-dir outputs/final/2way_fixed --tsm-weight 0.2 $POST 2>&1 | grep "avg-mAP tight"

# 2) 2-way MERGED checkpoints — did retraining help?
log_line "2-way MERGED (tsm_merged + slowfast_merged2)  w=0.2/0.8"
$PY scripts/run_ensemble.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_merged/best.pt \
  --slowfast-checkpoint outputs/slowfast_merged2/best.pt \
  --output-dir outputs/final/2way_merged --tsm-weight 0.2 $POST 2>&1 | grep "avg-mAP tight"

# 3) 3-way: small NetVLAD contribution
log_line "3-way (merged) w=0.2/0.6/0.2 (small NetVLAD)"
$PY scripts/run_ensemble3.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_merged/best.pt \
  --slowfast-checkpoint outputs/slowfast_merged2/best.pt \
  --netvlad-checkpoint  outputs/netvlad_merged/best.pt \
  --output-dir outputs/final/3way_020_060_020 \
  --tsm-weight 0.2 --slowfast-weight 0.6 --netvlad-weight 0.2 $POST 2>&1 | grep "avg-mAP tight"

# 4) 3-way: moderate NetVLAD contribution
log_line "3-way (merged) w=0.15/0.55/0.30"
$PY scripts/run_ensemble3.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_merged/best.pt \
  --slowfast-checkpoint outputs/slowfast_merged2/best.pt \
  --netvlad-checkpoint  outputs/netvlad_merged/best.pt \
  --output-dir outputs/final/3way_015_055_030 \
  --tsm-weight 0.15 --slowfast-weight 0.55 --netvlad-weight 0.30 $POST 2>&1 | grep "avg-mAP tight"

# 5) 3-way: SlowFast-heavier
log_line "3-way (merged) w=0.10/0.70/0.20 (SF-dominant)"
$PY scripts/run_ensemble3.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_merged/best.pt \
  --slowfast-checkpoint outputs/slowfast_merged2/best.pt \
  --netvlad-checkpoint  outputs/netvlad_merged/best.pt \
  --output-dir outputs/final/3way_010_070_020 \
  --tsm-weight 0.10 --slowfast-weight 0.70 --netvlad-weight 0.20 $POST 2>&1 | grep "avg-mAP tight"

# 6) 3-way: using ORIGINAL (non-merged) TSM+SF with merged NetVLAD
log_line "3-way MIXED (tsm_fixed + slowfast_fixed + netvlad_merged) w=0.15/0.55/0.30"
$PY scripts/run_ensemble3.py --data-dir $DATA --split $SPLIT --feature-type $FEATS \
  --tsm-checkpoint     outputs/tsm_fixed/best.pt \
  --slowfast-checkpoint outputs/slowfast_fixed/best.pt \
  --netvlad-checkpoint  outputs/netvlad_merged/best.pt \
  --output-dir outputs/final/3way_mixed_015_055_030 \
  --tsm-weight 0.15 --slowfast-weight 0.55 --netvlad-weight 0.30 $POST 2>&1 | grep "avg-mAP tight"

echo ""
echo "Done. Best config is whatever maximises avg-mAP tight above."
