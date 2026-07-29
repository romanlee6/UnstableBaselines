"""Phase-balanced fixed-seat multi-policy co-training for social dilemmas.

Select a game with ``UB_GAME=public_goods|stag_hunt|three_player_ipd`` and a
phase layout with ``UB_MODE=predict|broadcast``. Each player id owns an
independent LoRA adapter; adapters are trained concurrently without seat
shuffling or parameter sharing.
"""

import os
import pathlib
import re

import unstable


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"true", "1", "yes"}


GAME = os.environ.get("UB_GAME", "public_goods")
MODE = os.environ.get("UB_MODE", "predict")
if MODE not in {"predict", "broadcast"}:
    raise ValueError("UB_MODE must be 'predict' or 'broadcast'")

GAME_CONFIGS = {
    "public_goods": ("PublicGoodsGame", 3),
    "stag_hunt": ("IteratedStagHunt", 2),
    "three_player_ipd": ("ThreePlayerIPD", 3),
}
if GAME not in GAME_CONFIGS:
    raise ValueError(f"UB_GAME must be one of {sorted(GAME_CONFIGS)}")

env_prefix, num_players = GAME_CONFIGS[GAME]
variant = "Predict" if MODE == "predict" else "Broadcast"
env_id = f"{env_prefix}-{variant}-v0-train"
phases = (
    ("conversation", "prediction", "decision")
    if MODE == "predict"
    else ("conversation", "decision")
)
phase_samples = _int("UB_PHASE_SAMPLES", 128)
batch_size = phase_samples * len(phases)
learning_steps = _int("UB_LEARNING_STEPS", 100)

model_name = os.environ.get("UB_MODEL_NAME", "Qwen/Qwen3-4B-Base")
external_opponent = os.environ.get("AZURE_AI_DEPLOYMENT", "DeepSeek-V4-flash")
run_name = os.environ.get(
    "UB_RUN_NAME",
    f"UB-multirole-reinforce-{model_name.split('/')[-1]}-{GAME}-{MODE}-fixed-seat",
)
wandb_id = os.environ.get("UB_WANDB_ID")
resume_mode = os.environ.get("UB_RESUME_MODE", "auto").lower()
if resume_mode not in {"must", "auto", "never"}:
    raise ValueError("UB_RESUME_MODE must be one of: must, auto, never")

role_lora_config = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def _find_resume_state():
    output_roots = [pathlib.Path("outputs")]
    configured_root = pathlib.Path(
        os.path.expanduser(os.environ.get("UNSTABLE_OUTPUT_ROOT", "outputs"))
    )
    if configured_root not in output_roots:
        output_roots.append(configured_root)
    for root in os.environ.get("UNSTABLE_RESUME_ROOTS", "").split(os.pathsep):
        if root:
            candidate = pathlib.Path(os.path.expanduser(root))
            if candidate not in output_roots:
                output_roots.append(candidate)

    checkpoint_dirs = [
        path
        for output_root in output_roots
        for path in output_root.glob(f"*/*/{run_name}/checkpoints/iteration-*")
    ]

    def iteration(path):
        match = re.fullmatch(r"iteration-(\d+)", path.name)
        return int(match.group(1)) if match else -1

    def complete_role(path, pid):
        role_dir = path / f"role-{pid}"
        return (
            (role_dir / "adapter_config.json").is_file()
            and (role_dir / "adapter_model.safetensors").is_file()
        )

    complete = [
        path
        for path in checkpoint_dirs
        if all(complete_role(path, pid) for pid in range(num_players))
    ]
    if not complete:
        if resume_mode == "must":
            raise RuntimeError(f"No complete {num_players}-role checkpoint found for {run_name}")
        return 1, None, {pid: 0 for pid in range(num_players)}, None

    checkpoint = max(
        complete,
        key=lambda path: (iteration(path), path.stat().st_mtime),
    )
    checkpoint_step = iteration(checkpoint)
    role_steps = {
        pid: {
            iteration(path)
            for path in checkpoint_dirs
            if 0 < iteration(path) <= checkpoint_step and complete_role(path, pid)
        }
        for pid in range(num_players)
    }
    lora_paths = {
        pid: str(checkpoint / f"role-{pid}")
        for pid in range(num_players)
    }
    samples_seen = {
        pid: len(role_steps[pid]) * batch_size
        for pid in range(num_players)
    }
    training_state = checkpoint / "training_state.pt"
    return (
        checkpoint_step + 1,
        lora_paths,
        samples_seen,
        str(training_state) if training_state.is_file() else None,
    )


if resume_mode == "never":
    initial_step = 1
    initial_lora_paths = None
    initial_samples_seen = {pid: 0 for pid in range(num_players)}
    initial_training_state_path = None
else:
    (
        initial_step,
        initial_lora_paths,
        initial_samples_seen,
        initial_training_state_path,
    ) = _find_resume_state()

if initial_lora_paths:
    print(
        f"Resuming {run_name} at step {initial_step} from "
        f"{pathlib.Path(initial_lora_paths[0]).parent}",
        flush=True,
    )
else:
    print(f"Starting {run_name} from scratch for {learning_steps} steps", flush=True)


run = unstable.build_multirole(
    model_name=model_name,
    role_pids=list(range(num_players)),
    role_lora_cfgs={pid: role_lora_config for pid in range(num_players)},
    initial_lora_paths=initial_lora_paths,
    initial_step=initial_step,
    initial_samples_seen=initial_samples_seen,
    initial_training_state_path=initial_training_state_path,
    wandb_id=wandb_id,
    wandb_resume=("must" if initial_lora_paths else "allow"),
    run_name_override=run_name,
    shuffle_roles=False,
    train_envs=[
        unstable.TrainEnvSpec(
            env_id=env_id,
            num_players=num_players,
            num_actors=num_players,
            prompt_template="qwen3-multiphase",
        )
    ],
    eval_envs=[
        unstable.EvalEnvSpec(
            env_id=env_id,
            num_players=num_players,
            prompt_template="qwen3-multiphase",
            fixed_opponent=external_opponent,
        )
    ],
    eval_substitutions={
        pid: external_opponent for pid in range(1, num_players)
    },
    eval_provider=os.environ.get("UB_EVAL_PROVIDER", "azure_ai"),
    algorithm="reinforce",
    max_train_len=_int("UB_MAX_TRAIN_SEQ_LEN", 3000),
    max_generation_len=_int("UB_MAX_GENERATION_LENGTH", 256),
    retain_recent_context=True,
    batch_size=batch_size,
    mini_batch_size=_int("UB_MINI_BATCH_SIZE", 4),
    buffer_size=batch_size * 2,
    learning_rate=_float("UB_LEARNING_RATE", 1e-5),
    gradient_clipping=_float("UB_GRADIENT_CLIPPING", 0.2),
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False,
    env_step_reward_scale=1.0,
    payoff_reward_scale=_float("UB_PAYOFF_REWARD_SCALE", 1.0),
    prediction_reward_scale=_float(
        "UB_PREDICTION_REWARD_SCALE", 1.0 if MODE == "predict" else 0.0
    ),
    terminal_reward_scale=_float("UB_TERMINAL_REWARD_SCALE", 1.0),
    include_final_reward=_bool("UB_INCLUDE_FINAL_REWARD", False),
    include_invalid_move_reward=_bool("UB_INCLUDE_INVALID_MOVE_REWARD", False),
    balanced_phases=phases,
    phase_local_normalization=True,
    # Preserve the prediction experiment's absolute 1/3 coefficient. The
    # broadcast control therefore has total phase weight 2/3.
    phase_loss_weights={phase: 1.0 / 3.0 for phase in phases},
    normalize_phase_loss_weights=False,
    run_name_suffix=f"{GAME}-{MODE}-fixed-seat-multipolicy",
)

run.start(
    learning_steps=learning_steps + 1,
    num_collection_workers=_int("UB_COLLECTION_WORKERS", 384),
    num_eval_workers=_int("UB_EVAL_WORKERS", 8),
)
