#!/usr/bin/env bash
#SBATCH --job-name=ub-ipd-vllm-check
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/UnstableBaselines}"
INSTANCE_NAME="${INSTANCE_NAME:-vllm-ipd-generation-${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
export PROJECT_DIR INSTANCE_NAME
exec ./start_vllm_shell.sh bash tests/run_ipd_vllm_generation_check.sh
