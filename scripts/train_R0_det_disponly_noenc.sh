#!/bin/bash
#SBATCH --job-name=R0_det_disponly_noenc
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/R0_det_disponly_noenc_%j.out
#SBATCH --error=logs/R0_det_disponly_noenc_%j.err

# R0: DETERMINISTIC, disparity-only (1ch), NO semantic encoder. Counterpart of D0.
# Same network / data / schedule as the diffusion counterpart; the ONLY change is
# --deterministic: zero input at a fixed sigma, direct x0 regression (plain MSE, no noise).
# Data: data/stereo_128_{train,test} (left-view GT).
# Overrides:  END_STEP (default 10000000)  BATCH_SIZE (default 16)  RUN_NAME  EVAL_EVERY / DEMO_EVERY

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass

# Run from the repository root (sbatch scripts/<this script>).
mkdir -p logs

python train_udf.py \
  --config configs/foj_transformer_v2_stereo_R0_det_disponly_noenc_128_left.json \
  --batch-size ${BATCH_SIZE:-16} \
  --checkpointing \
  --evaluate-every ${EVAL_EVERY:-2000} \
  --num-workers 8 \
  --name ${RUN_NAME:-R0_det_disponly_noenc} \
  --demo-every ${DEMO_EVERY:-100000} \
  --end-step ${END_STEP:-10000000} \
  --deterministic
