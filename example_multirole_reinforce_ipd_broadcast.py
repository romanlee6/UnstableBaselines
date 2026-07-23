"""Multi-role REINFORCE on IteratedPrisonersDilemma-Broadcast-v0 (fixed pid).

Env config (registered in textarena/envs/__init__.py):
    - enable_broadcast_comm = True   (simultaneous {msg} broadcast, PGG-style)
    - enable_prediction     = False  (no prediction phase)
    - use_step_rewards      = True   (per-round payoff emitted via step_rewards_by_pid)
    - prediction_reward     = 0.0

Training config:
    - shuffle_roles = False  (pid IS role identity; adapter-{pid} always sits at env pid)
    - REINFORCE uses per-step rewards natively via StepBuffer + EnvStepReward, so the
      env's use_step_rewards flow lands as per-turn advantage without extra plumbing.
      (`use_turn_scores` is an a2c/ppo/grpo knob; REINFORCE has no equivalent toggle.)

wandb run name:
    UB-multirole-reinforce-Qwen3-4B-Base-IteratedPrisonersDilemma-Broadcast-v0-train-noshuffle-<ts>
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

run = unstable.build_multirole(
    model_name=MODEL_NAME,
    role_pids=[0, 1],
    role_lora_cfgs={0: ROLE_LORA_CFG, 1: ROLE_LORA_CFG},
    shuffle_roles=False,
    train_envs=[
        unstable.TrainEnvSpec(env_id="PublicGoodsGame-Broadcast-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_envs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-Broadcast-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
        unstable.EvalEnvSpec(env_id="PublicGoodsGame-Broadcast-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
    ],
    eval_substitutions={1: "google/gemini-3.1-flash-lite-preview-20260303"},
    algorithm="reinforce",
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH,
    batch_size=384,
    mini_batch_size=4,
    buffer_size=384 * 2,
    learning_rate=1e-5,
    gradient_clipping=0.2,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False,
    env_step_reward_scale=1.0,
    run_name_suffix="noshuffle",
)


run.start(learning_steps=400, num_collection_workers=384, num_eval_workers=16)
