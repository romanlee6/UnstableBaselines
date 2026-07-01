"""Asymmetric multi-role LoRA training example.

Two roles, each bound to its own pid. Both LoRAs share a single base model and
a single Ray learner actor (one GPU). vLLM serves both adapters concurrently
via per-request LoRARequest.

Eval substitutes pid=1 with an OpenRouter agent so role-0's LoRA is measured
against an unseen partner each eval game.
"""

import unstable
import unstable.reward_transformations as retra  # noqa: F401  (kept for parity with example_standard.py)

MODEL_NAME = "Qwen/Qwen3-4B-Base"
MAX_TRAIN_SEQ_LEN = 3000
MAX_GENERATION_LENGTH = 1024

# Match example_standard.py LoRA config (rank=32, all attention + MLP modules)
# so each role's adapter has the same capacity as the single-LoRA baseline.
ROLE_LORA_CFG = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

run = unstable.build_multirole(
    model_name=MODEL_NAME,
    role_pids=[0, 1],
    role_lora_cfgs={0: ROLE_LORA_CFG, 1: ROLE_LORA_CFG},
    train_envs=[
        unstable.TrainEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_envs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
        unstable.EvalEnvSpec(env_id="PublicGoodsGame-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
    ],
    # measure role-0's LoRA against an external partner each eval game
    eval_substitutions={1: "google/gemini-3.1-flash-lite-preview-20260303"},
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

run.start(learning_steps=100, num_collection_workers=128, num_eval_workers=8)
