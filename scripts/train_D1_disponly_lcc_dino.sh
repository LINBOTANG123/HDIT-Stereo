#!/bin/bash
#SBATCH --job-name=D1_disp_lcc_dino
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/D1_disp_lcc_dino_%j.out
#SBATCH --error=logs/D1_disp_lcc_dino_%j.err

# D1: diffusion, disparity-only (1ch), WITH semantic encoder (LCC + DINOv2). No UDF head.
# Data: generated_boundary_area_textureback_128_{train,test}_left (left-view GT).
# gpu_test only gives MIG-sliced A100s, so this runs single-process (no NCCL / multi-GPU).
# Overrides:  END_STEP (default 10000000)  BATCH_SIZE (default 16)  RUN_NAME  SAVE_EVERY / EVAL_EVERY / DEMO_EVERY
#   e.g. smoke test:  END_STEP=200 EVAL_EVERY=100 DEMO_EVERY=100000 RUN_NAME=D1_disp_lcc_dino_smoke sbatch scripts/train_D1_disponly_lcc_dino.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hourglass

cd /n/netscratch/zickler_lab/Lab/linbo/stereo_diffusion
mkdir -p logs

python train_udf.py \
  --config configs/foj_transformer_v2_stereo_D1_disponly_lcc_dino_128_left.json \
  --batch-size ${BATCH_SIZE:-16} \
  --checkpointing \
  --evaluate-every ${EVAL_EVERY:-2000} \
  --num-workers 8 \
  --name ${RUN_NAME:-D1_disp_lcc_dino} \
  --demo-every ${DEMO_EVERY:-100000} \
  --end-step ${END_STEP:-10000000} \
  --use-scc --scc-plane --scc-disp-weight 0.2 --scc-dmax 32 --disp-left
