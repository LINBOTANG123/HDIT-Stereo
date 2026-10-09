#!/bin/bash
# Train one of the four 128x128 (left-view GT) model variants.
#
#   bash scripts/train.sh <variant> [extra train_udf.py args...]
#
# variant:
#   baseline   plain denoiser, disparity only (1 channel)           [no cost volume]
#   dino       UDF + disparity, DINO cost volume only
#   lcc        UDF + disparity, Lean Correlation Conditioner (LCC) only
#   lcc_dino   UDF + disparity, LCC + DINO cost volume              [full model]
#
# Run from the repository root, after generating data into data/stereo_128_{train,test}
# (see scripts/generate_data.sh). Environment overrides:
#   NUM_GPUS (default 4)   BATCH_SIZE (default 16, per process)   NUM_WORKERS (default 8)
#   PORT (default 29511)   RUN_NAME (default hdit_stereo_<variant>_128_left)
#   WANDB_PROJECT / WANDB_ENTITY   enable Weights & Biases logging when WANDB_PROJECT is set
set -euo pipefail

VARIANT="${1:?usage: bash scripts/train.sh <baseline|dino|lcc|lcc_dino> [extra args]}"
shift

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PORT="${PORT:-29511}"
RUN_NAME="${RUN_NAME:-hdit_stereo_${VARIANT}_128_left}"

# Flags shared by every UDF+disparity variant (everything except the baseline).
UDF_COMMON=(--udf-channel-weight 1.0 --fg-alpha 0.0 --udf-k 3.0
            --use-scc --scc-plane --scc-disp-weight 0.2 --scc-dmax 32 --disp-left)

case "$VARIANT" in
  baseline) CONFIG=configs/foj_transformer_v2_stereo_baseline_plain_disponly_128_left.json; VARIANT_ARGS=() ;;
  dino)     CONFIG=configs/foj_transformer_v2_stereo_dino_cv_dispudf_128_left.json;         VARIANT_ARGS=("${UDF_COMMON[@]}" --scc-dino-only) ;;
  lcc)      CONFIG=configs/foj_transformer_v2_stereo_lcc_dispudf_128_left.json;             VARIANT_ARGS=("${UDF_COMMON[@]}") ;;
  lcc_dino) CONFIG=configs/foj_transformer_v2_stereo_lcc_dino_dispudf_128_left.json;        VARIANT_ARGS=("${UDF_COMMON[@]}") ;;
  *) echo "unknown variant '$VARIANT' (expected baseline|dino|lcc|lcc_dino)" >&2; exit 1 ;;
esac

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS=(--wandb-project "$WANDB_PROJECT")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

mkdir -p logs
accelerate launch --num_processes "$NUM_GPUS" --main_process_ip 127.0.0.1 --main_process_port "$PORT" \
  --mixed_precision bf16 train_udf.py \
  --config "$CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --checkpointing \
  --evaluate-every 2000 \
  --num-workers "$NUM_WORKERS" \
  --name "$RUN_NAME" \
  --demo-every 100000 \
  --end-step 10000000 \
  "${VARIANT_ARGS[@]}" "${WANDB_ARGS[@]}" "$@"
