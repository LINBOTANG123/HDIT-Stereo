#!/bin/bash
# Aggregate metrics for one inference output directory.
#
#   bash scripts/eval_metrics.sh <PRED_DIR> [GT_DIR]
#
# PRED_DIR is an infer.py --out-dir (contains metrics.csv + per-sample *_disp.npy).
# GT_DIR   defaults to data/stereo_128_test (needs disp_left/ and, for shape recovery,
#          amodal_mask_obj{1,2}/ + disp_obj{1,2}/).
#
# Writes into PRED_DIR:  boundary_metrics.csv (sharpness / boundary-vs-interior EPE),
#                        agg.csv (per-seed mean/median/best-K + map-median ensemble),
#                        shape_recovery.csv (shapematch IoU per object).
set -euo pipefail

PRED="${1:?usage: bash scripts/eval_metrics.sh <PRED_DIR> [GT_DIR]}"
GT="${2:-data/stereo_128_test}"

python report_metrics.py boundary --pred-dir "$PRED" --gt-dir "$GT/disp_left" \
  --out-csv "$PRED/boundary_metrics.csv"
python report_metrics.py csv "$PRED" --best-k 1 3 --ensemble --boundary --out-csv "$PRED/agg.csv"
python shape_recovery_eval.py --gt-dir "$GT" --pred-dir "model=$PRED" --out-csv "$PRED/shape_recovery.csv"
