#!/usr/bin/env bash
#SBATCH --job-name=ub-ipd-transaction-test
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/UnstableBaselines}"
INSTANCE_NAME="${INSTANCE_NAME:-vllm-ipd-transaction-test-${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
export PROJECT_DIR INSTANCE_NAME
exec ./start_vllm_shell.sh bash tests/run_ipd_transactional_checks.sh
