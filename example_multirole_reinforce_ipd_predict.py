"""Multi-role REINFORCE with opponent-action prediction on IPD.

The registered ``IteratedPrisonersDilemma-Predict-v0-train`` environment runs
three queries per player in each round:

1. simultaneous ``[Message: ...]`` communication,
2. private ``[Prediction: Cooperate]`` / ``[Prediction: Defect]`` prediction,
3. simultaneous ``[Action: Cooperate]`` / ``[Action: Defect]`` decision.

The shared UB prompt still places each command inside ``\\boxed{...}``; the
environment receives and validates the extracted square-bracket payload.

The collector routes the prediction bonus back to the prediction completion and
the game payoff to the decision completion. Both role adapters train in
self-play; seats remain fixed so role 0 always maps to player 0.
"""

import os
import pathlib
import re

import unstable


MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_TRAIN_SEQ_LEN = int(os.environ.get("UB_MAX_TRAIN_SEQ_LEN", "3000"))
MAX_GENERATION_LENGTH = int(os.environ.get("UB_MAX_GENERATION_LENGTH", "256"))
BATCH_SIZE = 384
AZURE_EVAL_DEPLOYMENT = os.environ.get("AZURE_AI_DEPLOYMENT", "DeepSeek-V4-flash")

ROLE_LORA_CFG = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

RUN_NAME = os.environ.get(
    "UB_RUN_NAME",
    "UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Predict-v0-train-ipd-predict-noshuffle-1783995598",
)
WANDB_ID = os.environ.get("UB_WANDB_ID", "f8shgd49")
RESUME_MODE = os.environ.get("UB_RESUME_MODE", "must").lower()
LEARNING_STEPS = int(os.environ.get("UB_LEARNING_STEPS", "300"))


def find_resume_state() -> tuple[int, dict[int, str] | None, dict[int, int], str | None]:
    """Find the newest complete two-role checkpoint for this interrupted run.

    A preemption can happen between role updates. Only an iteration containing
    both complete adapters is safe to load, so a partial newer iteration is
    intentionally ignored. Samples seen are reconstructed from unique saved
    role/iteration pairs, which also handles this run's earlier partial step 9.
    """
    # Search both the historical repo-local output tree and the configured
    # scratch/pool tree. This lets the first requeue migrate locations without
    # losing sight of its last home-directory checkpoint.
    output_roots = [pathlib.Path("outputs")]
    configured_root = pathlib.Path(os.path.expanduser(os.environ.get("UNSTABLE_OUTPUT_ROOT", "outputs")))
    if configured_root not in output_roots:
        output_roots.append(configured_root)
    for root in os.environ.get("UNSTABLE_RESUME_ROOTS", "").split(os.pathsep):
        if root:
            resume_root = pathlib.Path(os.path.expanduser(root))
            if resume_root not in output_roots:
                output_roots.append(resume_root)
    checkpoint_dirs = [
        path
        for output_root in output_roots
        for path in output_root.glob(f"*/*/{RUN_NAME}/checkpoints/iteration-*")
    ]

    def iteration(path: pathlib.Path) -> int:
        match = re.fullmatch(r"iteration-(\d+)", path.name)
        return int(match.group(1)) if match else -1

    def complete_adapter(path: pathlib.Path, pid: int) -> bool:
        role_dir = path / f"role-{pid}"
        return (
            (role_dir / "adapter_config.json").is_file()
            and (role_dir / "adapter_model.safetensors").is_file()
        )

    complete = [
        path for path in checkpoint_dirs
        if complete_adapter(path, 0) and complete_adapter(path, 1)
    ]
    if not complete and RESUME_MODE == "auto":
        return 1, None, {0: 0, 1: 0}, None
    if not complete:
        raise RuntimeError(f"No complete two-role checkpoint found for {RUN_NAME}")

    checkpoint_dir = max(complete, key=lambda path: (iteration(path), path.stat().st_mtime))
    checkpoint_step = iteration(checkpoint_dir)
    role_steps = {
        pid: {
            iteration(path)
            for path in checkpoint_dirs
            if 0 < iteration(path) <= checkpoint_step and complete_adapter(path, pid)
        }
        for pid in (0, 1)
    }
    lora_paths = {pid: str(checkpoint_dir / f"role-{pid}") for pid in (0, 1)}
    samples_seen = {pid: len(role_steps[pid]) * BATCH_SIZE for pid in (0, 1)}
    training_state = checkpoint_dir / "training_state.pt"
    return checkpoint_step + 1, lora_paths, samples_seen, (
        str(training_state) if training_state.is_file() else None
    )


if RESUME_MODE not in {"must", "auto", "never"}:
    raise ValueError("UB_RESUME_MODE must be one of: must, auto, never")
if RESUME_MODE == "never":
    INITIAL_STEP, INITIAL_LORA_PATHS = 1, None
    INITIAL_SAMPLES_SEEN, INITIAL_TRAINING_STATE_PATH = {0: 0, 1: 0}, None
else:
    INITIAL_STEP, INITIAL_LORA_PATHS, INITIAL_SAMPLES_SEEN, INITIAL_TRAINING_STATE_PATH = find_resume_state()

if INITIAL_LORA_PATHS:
    print(
        "Resuming", RUN_NAME, f"at step {INITIAL_STEP}",
        f"from {pathlib.Path(INITIAL_LORA_PATHS[0]).parent}",
        f"with samples_seen={INITIAL_SAMPLES_SEEN}",
        f"training_state={INITIAL_TRAINING_STATE_PATH or 'legacy weights-only checkpoint'}",
        flush=True,
    )
else:
    print("Starting from scratch", RUN_NAME, f"for {LEARNING_STEPS} steps", flush=True)

run = unstable.build_multirole(
    model_name=MODEL_NAME,
    role_pids=[0, 1],
    role_lora_cfgs={0: ROLE_LORA_CFG, 1: ROLE_LORA_CFG},
    initial_lora_paths=INITIAL_LORA_PATHS,
    initial_step=INITIAL_STEP,
    initial_samples_seen=INITIAL_SAMPLES_SEEN,
    initial_training_state_path=INITIAL_TRAINING_STATE_PATH,
    wandb_id=WANDB_ID,
    wandb_resume=("must" if INITIAL_LORA_PATHS else "allow"),
    run_name_override=RUN_NAME,
    shuffle_roles=False,
    train_envs=[
        unstable.TrainEnvSpec(
            env_id="IteratedPrisonersDilemma-Predict-v0-train",
            num_players=2,
            num_actors=2,
            prompt_template="qwen3-multiphase",
        ),
    ],
    # Evaluate role 0 against a fixed external opponent; both roles continue
    # to be trained in the self-play environment above.
    eval_envs=[
        unstable.EvalEnvSpec(
            env_id="IteratedPrisonersDilemma-Predict-v0-train",
            num_players=2,
            prompt_template="qwen3-multiphase",
            fixed_opponent=AZURE_EVAL_DEPLOYMENT,
        ),
    ],
    eval_substitutions={1: AZURE_EVAL_DEPLOYMENT},
    eval_provider="azure_ai",
    algorithm="reinforce",
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH,
    retain_recent_context=True,
    batch_size=BATCH_SIZE,
    mini_batch_size=4,
    buffer_size=BATCH_SIZE * 2,
    learning_rate=1e-5,
    gradient_clipping=0.2,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False,
    # Keep both decision payoff and prediction-accuracy reward in the update.
    env_step_reward_scale=1.0,
    # Phase-isolated REINFORCE: communication learns format, prediction learns
    # format+accuracy, and decision learns format+round payoff. Terminal game
    # outcome and the legacy non-invalid-move bonus are intentionally excluded.
    include_final_reward=False,
    include_invalid_move_reward=False,
    balanced_phases=("conversation", "prediction", "decision"),
    phase_local_normalization=True,
    phase_loss_weights={"conversation": 1.0, "prediction": 1.0, "decision": 1.0},
    run_name_suffix="ipd-predict-noshuffle",
)

# Requires one learner GPU plus collector/vLLM GPUs.
run.start(learning_steps=LEARNING_STEPS, num_collection_workers=384, num_eval_workers=8)
