#!/bin/bash
#SBATCH --job-name=distillation
#SBATCH --output=logs/%j_distill.out
#SBATCH --error=logs/%j_distill.err
#SBATCH --partition=h200          # update to your cluster's H200 partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load cuda/12.4               # update to match your cluster's CUDA module

source activate distill             # or: conda activate distill / source venv/bin/activate

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# ── Run ───────────────────────────────────────────────────────────────────────
python distill.py

echo "End time: $(date)"
