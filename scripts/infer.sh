#!/bin/bash
# Run multi-seed inference with a trained checkpoint on a generated test split.
#
#   bash scripts/infer.sh <checkpoint.pth> [DATA_DIR] [OUT_DIR]
#
# DATA_DIR defaults to data/stereo_128_test (left/, right/, disp_left/, fields_left/).
# OUT_DIR  defaults to results/<checkpoint basename>.
# Environment overrides: SEEDS (default "0 1 2 3 4 5 6 7 8 9"), STEPS (default 50), GLOB.
#
# infer.py reads the model config from the checkpoint. The UDF decode (exponential vs
# linear) and the disparity-only (1-channel) case are inferred from that config.
set -euo pipefail

CKPT="${1:?usage: bash scripts/infer.sh <checkpoint.pth> [DATA_DIR] [OUT_DIR]}"
DATA="${2:-data/stereo_128_test}"
OUT="${3:-results/$(basename "$CKPT" .pth)}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
STEPS="${STEPS:-50}"
GLOB="${GLOB:-*_left.png}"

# shellcheck disable=SC2086
python infer.py \
  --checkpoint "$CKPT" \
  --left-dir "$DATA/left" --right-dir "$DATA/right" \
  --gt-field-dir "$DATA/fields_left" --gt-disp-dir "$DATA/disp_left" \
  --glob "$GLOB" \
  --seeds $SEEDS --steps "$STEPS" \
  --out-dir "$OUT"
