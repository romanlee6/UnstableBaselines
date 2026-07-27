#!/usr/bin/env bash
#SBATCH --job-name=ub-ipd-typed-recent
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gpus=rtx_pro_6000:4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/UnstableBaselines}"
INSTANCE_NAME="${INSTANCE_NAME:-vllm-${SLURM_JOB_ID}}"
UNSTABLE_OUTPUT_ROOT="${UNSTABLE_OUTPUT_ROOT:-$HOME/orcd/scratch/UnstableBaselines/outputs}"

# With no checkpoint this starts from the bare Qwen3-4B base model. If Slurm
# requeues the job after a preemption, it resumes the newest complete checkpoint.
UB_RUN_NAME="UB-multirole-reinforce-Qwen3-4B-Base-IPD-Predict-typed-recent-context-${SLURM_JOB_ID}"
UB_WANDB_ID="ub-ipd-typed-recent-${SLURM_JOB_ID}"
UB_RESUME_MODE="auto"
UB_LEARNING_STEPS="${UB_LEARNING_STEPS:-200}"
UB_ERROR_ALLOWANCE="${UB_ERROR_ALLOWANCE:-2}"
UB_MAX_TRAIN_SEQ_LEN="${UB_MAX_TRAIN_SEQ_LEN:-3000}"
UB_MAX_GENERATION_LENGTH="${UB_MAX_GENERATION_LENGTH:-256}"

AZURE_AI_DEPLOYMENT="${AZURE_AI_DEPLOYMENT:-DeepSeek-V4-flash}"
UB_EVAL_MAX_TOKENS="${UB_EVAL_MAX_TOKENS:-1024}"
UB_EVAL_TEMPERATURE="${UB_EVAL_TEMPERATURE:-0.0}"

if [[ -z "${AZURE_AI_API_KEY:-${AZURE_OPENAI_API_KEY:-${AZURE_INFERENCE_CREDENTIAL:-${ANTHROPIC_FOUNDRY_API_KEY:-}}}}" ]]; then
  echo "Missing Azure API key environment variable." >&2
  exit 2
fi
if [[ -z "${AZURE_AI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT:-${AZURE_FOUNDRY_ENDPOINT:-${AZURE_AI_RESOURCE:-${AZURE_OPENAI_RESOURCE:-${ANTHROPIC_FOUNDRY_RESOURCE:-}}}}}}" ]]; then
  echo "Missing Azure endpoint or resource environment variable." >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p "$UNSTABLE_OUTPUT_ROOT"

echo "[$(date --iso-8601=seconds)] Fresh typed-IPD run on $(hostname), Slurm job ${SLURM_JOB_ID}"
echo "Run: ${UB_RUN_NAME}; base model only; resume mode: ${UB_RESUME_MODE}"
echo "Context: train=${UB_MAX_TRAIN_SEQ_LEN}, generation=${UB_MAX_GENERATION_LENGTH}"

export PROJECT_DIR INSTANCE_NAME UNSTABLE_OUTPUT_ROOT
export UB_RUN_NAME UB_WANDB_ID UB_RESUME_MODE UB_LEARNING_STEPS UB_ERROR_ALLOWANCE
export UB_MAX_TRAIN_SEQ_LEN UB_MAX_GENERATION_LENGTH
export AZURE_AI_DEPLOYMENT UB_EVAL_MAX_TOKENS UB_EVAL_TEMPERATURE

exec ./start_vllm_shell.sh python -u example_multirole_reinforce_ipd_predict.py
