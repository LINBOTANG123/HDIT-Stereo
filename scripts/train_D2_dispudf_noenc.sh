#!/bin/bash
#SBATCH --job-name=D2_dispudf_noenc
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/D2_dispudf_noenc_%j.out
#SBATCH --error=logs/D2_dispudf_noenc_%j.err

# D2: diffusion, UDF+disparity (2ch), NO semantic encoder (no LCC, no DINO).
# Data: data/stereo_128_{train,test} (left-view GT).
# gpu_test only gives MIG-sliced A100s, so this runs single-process (no NCCL / multi-GPU).
# Overrides:  END_STEP (default 10000000)  BATCH_SIZE (default 16)  RUN_NAME  SAVE_EVERY / EVAL_EVERY / DEMO_EVERY
#   e.g. smoke test:  END_STEP=200 EVAL_EVERY=100 DEMO_EVERY=100000 RUN_NAME=D2_dispudf_noenc_smoke sbatch scripts/train_D2_dispudf_noenc.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass

# Run from the repository root (sbatch scripts/<this script>).
mkdir -p logs

python train_udf.py \
  --config configs/foj_transformer_v2_stereo_D2_dispudf_noenc_128_left.json \
  --batch-size ${BATCH_SIZE:-16} \
  --checkpointing \
  --evaluate-every ${EVAL_EVERY:-2000} \
  --num-workers 8 \
  --name ${RUN_NAME:-D2_dispudf_noenc} \
  --demo-every ${DEMO_EVERY:-100000} \
  --end-step ${END_STEP:-10000000} \
  --udf-channel-weight 1.0 --fg-alpha 0.0 --udf-k 3.0
