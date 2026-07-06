"""Multi-role LoRA training on IPD with symmetric pid shuffling.

Two role LoRAs share one base model. `shuffle_roles=True` in the team sampler
randomizes which role's LoRA sits at env-pid=0 vs env-pid=1 per training
episode, mitigating IPD's leaky-conversation asymmetry (pid=1 sees pid=0's
message before responding). Trajectories are still routed to the correct
per-role buffer via `AgentSpec.role_pid` (decoupled from env pid).

Eval keeps a deterministic seat assignment: pid=1 is replaced by an
OpenRouter partner, and role-0's LoRA is measured against it at pid=0.
"""

import unstable
import unstable.reward_transformations as retra  # noqa: F401

MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_TRAIN_SEQ_LEN = 3000
MAX_GENERATION_LENGTH = 1024

ROLE_LORA_CFG = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Resume from prior run's iteration-87 (last iter where both role adapters were fully written).
RESUME_CKPT_DIR = "outputs/2026-07-02/16-40-18/UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-v0-train-1783024814/checkpoints/iteration-87"
INITIAL_LORA_PATHS = {0: f"{RESUME_CKPT_DIR}/role-0", 1: f"{RESUME_CKPT_DIR}/role-1"}

run = unstable.build_multirole(
    model_name=MODEL_NAME,
    role_pids=[0, 1],
    role_lora_cfgs={0: ROLE_LORA_CFG, 1: ROLE_LORA_CFG},
    initial_lora_paths=INITIAL_LORA_PATHS,
    shuffle_roles=True,
    train_envs=[
        unstable.TrainEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_envs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
        unstable.EvalEnvSpec(env_id="PublicGoodsGame-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
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
)

# 1 learner GPU + 2 collector GPUs. Set CUDA_VISIBLE_DEVICES to 3 GPUs before
# launching; Ray auto-assigns the learner one and the collector will spawn one
# vLLM actor per remaining GPU.
run.start(learning_steps=200, num_collection_workers=128, num_eval_workers=8)
