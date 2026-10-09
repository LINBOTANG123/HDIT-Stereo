#!/bin/bash
#SBATCH --job-name=R1_det_disponly_lcc_dino
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/R1_det_disponly_lcc_dino_%j.out
#SBATCH --error=logs/R1_det_disponly_lcc_dino_%j.err

# R1: DETERMINISTIC, disparity-only (1ch), WITH semantic encoder (LCC + DINOv2). Counterpart of D1.
# Same network / data / conditioning / schedule as its diffusion counterpart; the ONLY change is
# --deterministic: zero input at a fixed sigma, direct x0 regression (plain MSE, no noise).
# Inference: python infer.py ... (reads the flag from the checkpoint; one forward pass, use a single seed).
# Data: generated_boundary_area_textureback_128_{train,test}_left (left-view GT).
# Overrides:  END_STEP (default 10000000)  BATCH_SIZE (default 16)  RUN_NAME  EVAL_EVERY / DEMO_EVERY

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass

cd /n/netscratch/zickler_lab/Lab/linbo/stereo_diffusion
mkdir -p logs

python train_udf.py \
  --config configs/foj_transformer_v2_stereo_R1_det_disponly_lcc_dino_128_left.json \
  --batch-size ${BATCH_SIZE:-16} \
  --checkpointing \
  --evaluate-every ${EVAL_EVERY:-2000} \
  --num-workers 8 \
  --name ${RUN_NAME:-R1_det_disponly_lcc_dino} \
  --demo-every ${DEMO_EVERY:-100000} \
  --end-step ${END_STEP:-10000000} \
  --deterministic \
  --use-scc --scc-plane --scc-disp-weight 0.2 --scc-dmax 32 --disp-left
