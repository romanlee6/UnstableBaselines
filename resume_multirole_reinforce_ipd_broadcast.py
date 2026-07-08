"""Resume the crashed multi-role REINFORCE run.

Original run:
    UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Broadcast-v0-train-noshuffle-1783457248
    started 2026-07-07 16:47:33, crashed 2026-07-07 ~20:00:48 mid-step-32 (SIGKILL, no traceback).
    Last saved checkpoint: iteration-31.  wandb id: asl7pl50.

Continues the same wandb run (id=asl7pl50, resume="allow") so the learner/step
axis is contiguous: this run starts logging at learner/step=32.
"""

import os

import unstable
import unstable.reward_transformations as retra  # noqa: F401

MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_TRAIN_SEQ_LEN = 3000
MAX_GENERATION_LENGTH = 1024

ROLE_LORA_CFG = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

_ORIG_RUN_DIR = "outputs/2026-07-07/16-47-33/UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Broadcast-v0-train-noshuffle-1783457248"
_LAST_ITER = 31
INITIAL_LORA_PATHS = {
    0: os.path.abspath(f"{_ORIG_RUN_DIR}/checkpoints/iteration-{_LAST_ITER}/role-0"),
    1: os.path.abspath(f"{_ORIG_RUN_DIR}/checkpoints/iteration-{_LAST_ITER}/role-1"),
}
for _p in INITIAL_LORA_PATHS.values():
    assert os.path.isfile(os.path.join(_p, "adapter_config.json")), f"missing LoRA at {_p}"

ORIG_RUN_NAME = "UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Broadcast-v0-train-noshuffle-1783457248"
WANDB_ID = "asl7pl50"

run = unstable.build_multirole(
    model_name=MODEL_NAME,
    role_pids=[0, 1],
    role_lora_cfgs={0: ROLE_LORA_CFG, 1: ROLE_LORA_CFG},
    shuffle_roles=False,
    train_envs=[
        unstable.TrainEnvSpec(env_id="IteratedPrisonersDilemma-Broadcast-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_envs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-Broadcast-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
    ],
    eval_substitutions={1: "google/gemini-3.1-flash-lite-preview-20260303"},
    algorithm="reinforce",
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH,
    batch_size=384,
    mini_batch_size=1,
    buffer_size=384 * 2,
    learning_rate=1e-5,
    gradient_clipping=0.2,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False,
    env_step_reward_scale=1.0,
    run_name_suffix="noshuffle",
    initial_lora_paths=INITIAL_LORA_PATHS,
    initial_step=_LAST_ITER + 1,
    wandb_id=WANDB_ID,
    wandb_resume="allow",
    run_name_override=ORIG_RUN_NAME,
)

run.start(learning_steps=200, num_collection_workers=128, num_eval_workers=8)
