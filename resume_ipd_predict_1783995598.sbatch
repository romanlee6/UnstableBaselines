#!/usr/bin/env bash
#SBATCH --job-name=ub-ipd-predict-phase-balanced
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gpus=rtx_pro_6000:4
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/UnstableBaselines}"
INSTANCE_NAME="${INSTANCE_NAME:-vllm-${SLURM_JOB_ID}}"
UNSTABLE_OUTPUT_ROOT="${UNSTABLE_OUTPUT_ROOT:-$HOME/orcd/scratch/UnstableBaselines/outputs}"
UNSTABLE_RESUME_ROOTS="${UNSTABLE_RESUME_ROOTS:-$HOME/orcd/pool/UnstableBaselines/outputs}"
UB_RUN_NAME="${UB_RUN_NAME:-UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Predict-v0-train-phase-balanced-${SLURM_JOB_ID}}"
UB_WANDB_ID="${UB_WANDB_ID:-ub-ipd-predict-${SLURM_JOB_ID}}"
UB_RESUME_MODE="${UB_RESUME_MODE:-auto}"
UB_LEARNING_STEPS="${UB_LEARNING_STEPS:-300}"
UB_ERROR_ALLOWANCE="${UB_ERROR_ALLOWANCE:-2}"

cd "$PROJECT_DIR"
mkdir -p "$UNSTABLE_OUTPUT_ROOT"

echo "[$(date --iso-8601=seconds)] Starting on $(hostname) as Slurm job ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}; restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Run: ${UB_RUN_NAME}; target steps: ${UB_LEARNING_STEPS}; resume mode: ${UB_RESUME_MODE}"
echo "Malformed-action retry allowance: ${UB_ERROR_ALLOWANCE}"

export PROJECT_DIR INSTANCE_NAME UNSTABLE_OUTPUT_ROOT UNSTABLE_RESUME_ROOTS
export UB_RUN_NAME UB_WANDB_ID UB_RESUME_MODE UB_LEARNING_STEPS UB_ERROR_ALLOWANCE
exec ./start_vllm_shell.sh python -u example_multirole_reinforce_ipd_predict.py
