#!/usr/bin/env bash

set -euo pipefail

INSTANCE_NAME="${INSTANCE_NAME:-vllm}"
IMAGE_PATH="${IMAGE_PATH:-$HOME/vllm-openai.sif}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/UnstableBaselines}"

# ---------------------------------------------------------------------------
# Storage tiers (Engaging cluster).
#
# home         (NFS, backed up):    source of truth for code and requirements.
# orcd/scratch (NFS, 1 TB, persists across SLURM jobs):
#                                   active training outputs/checkpoints and HF
#                                   model cache. Persistent, but not backed up.
# /scratch     (node-local XFS, wiped when SLURM releases the node):
#                                   venv, apptainer tmp/cache, Ray tmp.
#                                   ONLY derivable data — nothing critical.
#
# ---------------------------------------------------------------------------
LOCAL_SCRATCH="/scratch/${USER}/ub"
CLUSTER_SCRATCH="${HOME}/orcd/scratch/ub"
DEFAULT_OUTPUT_ROOT="${HOME}/orcd/scratch/UnstableBaselines/outputs"

LOCAL_VENV="${LOCAL_SCRATCH}/venv"
HF_CACHE_DIR="${CLUSTER_SCRATCH}/hf"

mkdir -p "$LOCAL_SCRATCH" "$CLUSTER_SCRATCH" "$HF_CACHE_DIR" \
         "$LOCAL_SCRATCH/apptainer_cache" "$LOCAL_SCRATCH/apptainer_tmp"

export APPTAINER_CACHEDIR="$LOCAL_SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$LOCAL_SCRATCH/apptainer_tmp"

# ---------------------------------------------------------------------------
# Module init.
# ---------------------------------------------------------------------------
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/Modules/init/bash
else
  echo "Could not initialize the environment modules command." >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  echo "The environment modules command is unavailable after initialization." >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Missing Apptainer image: $IMAGE_PATH" >&2
  exit 1
fi

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  echo "Missing source virtualenv directory: $PROJECT_DIR/.venv" >&2
  echo "The home-side .venv is the source of truth. Install once before running." >&2
  exit 1
fi

loaded_modules="$(module -t list 2>&1 || true)"
if [[ "$loaded_modules" == *"gcc/12.2.0"* ]]; then
  module unload gcc/12.2.0
fi

module load apptainer

# ---------------------------------------------------------------------------
# Hydrate the venv onto node-local /scratch.
# One-time per SLURM node allocation. Python imports touch ~10k small files;
# on NFS home that costs many minutes, on local /scratch it costs milliseconds.
# ---------------------------------------------------------------------------
if [[ ! -x "$LOCAL_VENV/bin/python" ]]; then
  echo "[start_vllm_shell] Hydrating venv $PROJECT_DIR/.venv -> $LOCAL_VENV"
  echo "[start_vllm_shell] (one-time cost per compute node; ~2-5 min)"
  rsync -a --delete "$PROJECT_DIR/.venv/" "$LOCAL_VENV/"
  # Patch hard-coded venv paths in shebangs, activate scripts, and pyvenv.cfg.
  find "$LOCAL_VENV/bin" -type f -print0 2>/dev/null \
    | xargs -0 grep -Il "$PROJECT_DIR/\.venv" 2>/dev/null \
    | xargs -r sed -i "s|$PROJECT_DIR/.venv|$LOCAL_VENV|g" || true
  [[ -f "$LOCAL_VENV/pyvenv.cfg" ]] && \
    sed -i "s|$PROJECT_DIR/.venv|$LOCAL_VENV|g" "$LOCAL_VENV/pyvenv.cfg"
fi

# ---------------------------------------------------------------------------
# Instance start. --bind /scratch is required — apptainer does not include
# /scratch by default, so the container cannot see the local venv without it.
# ---------------------------------------------------------------------------
instance_exists=false
while IFS= read -r line; do
  [[ -z "$line" || "$line" == INSTANCE\ NAME* ]] && continue
  if [[ "$line" == "$INSTANCE_NAME"* ]]; then
    instance_exists=true
    break
  fi
done < <(apptainer instance list "$INSTANCE_NAME" 2>/dev/null || true)

if [[ "$instance_exists" != true ]]; then
  # ~/orcd/scratch is a symlink into /orcd/scratch/orcd/NNN/$USER (autofs NFS).
  # Apptainer does not propagate autofs mounts into the container, so we
  # explicitly bind the resolved target.
  CLUSTER_SCRATCH_REAL="$(readlink -f "$HOME/orcd/scratch")"
  apptainer instance start --cleanenv --nv \
    --bind /scratch \
    --bind "$CLUSTER_SCRATCH_REAL" \
    "$IMAGE_PATH" "$INSTANCE_NAME"
fi

# ---------------------------------------------------------------------------
# Env vars visible inside the container (APPTAINERENV_* -> plain name inside).
# HF_HOME points at persistent cluster scratch so model weights survive
# eviction and don't need to be re-downloaded.
# ---------------------------------------------------------------------------
export APPTAINERENV_HF_HOME="$HF_CACHE_DIR"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export APPTAINERENV_LOCAL_VENV="$LOCAL_VENV"
export APPTAINERENV_PROJECT_DIR="$PROJECT_DIR"
export APPTAINERENV_UNSTABLE_OUTPUT_ROOT="${UNSTABLE_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
export APPTAINERENV_UNSTABLE_RESUME_ROOTS="${UNSTABLE_RESUME_ROOTS:-}"
# Fixed-opponent credentials/configuration are opt-in passthroughs. The
# container is started with --cleanenv, so Azure evaluation cannot see these
# host variables unless they are explicitly mapped into the instance.
for azure_var in \
  AZURE_AI_API_KEY AZURE_AI_ENDPOINT AZURE_AI_RESOURCE AZURE_AI_DEPLOYMENT \
  AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AZURE_OPENAI_RESOURCE \
  AZURE_INFERENCE_CREDENTIAL AZURE_FOUNDRY_ENDPOINT \
  ANTHROPIC_FOUNDRY_API_KEY ANTHROPIC_FOUNDRY_RESOURCE \
  UB_EVAL_MAX_TOKENS UB_EVAL_TEMPERATURE; do
  if [[ -n "${!azure_var:-}" ]]; then
    export "APPTAINERENV_${azure_var}=${!azure_var}"
  fi
done
# Prefer the repository's vendored TextArena over the package installed in the
# virtualenv.  The local copy contains UB-specific environments and registrations
# (for example *-Predict-v0 and *-Broadcast-v0) that upstream PyPI TextArena does
# not provide.
export APPTAINERENV_PYTHONPATH="$PROJECT_DIR/third_party/TextArena:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -gt 0 ]]; then
  apptainer exec "instance://$INSTANCE_NAME" bash -c "cd \"\$PROJECT_DIR\" && source \"\$LOCAL_VENV/bin/activate\" && exec \"\$@\"" bash "$@"
else
  apptainer exec "instance://$INSTANCE_NAME" bash -ic "cd \"\$PROJECT_DIR\" && source \"\$LOCAL_VENV/bin/activate\" && exec bash -i"
fi
