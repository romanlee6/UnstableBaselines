"""Vanilla GRPO test on IteratedPrisonersDilemma.

Self-play multirole: two LoRAs on a shared base model, one per pid. GRPO
subtracts a per-(env_id, pid) group baseline at update time, so we drop the
RoleAdvantageByEnvFormatter EMA baseline automatically inside build_multirole.

"""
import os
import unstable

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
    shuffle_roles=True,
    train_envs=[
        unstable.TrainEnvSpec(env_id="PublicGoodsGame-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
    ],
    eval_envs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
        unstable.EvalEnvSpec(env_id="PublicGoodsGame-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
    ],
    eval_substitutions={1: "google/gemini-3.1-flash-lite-preview-20260303"},

    algorithm="grpo",
    use_turn_scores=True,
    n_epochs=2,
    clip_eps=0.2,
    kl_loss_coef=0.0,            # vanilla GRPO -> no KL penalty
    normalize_adv_by_pid=False,  # vanilla GRPO baseline only; no extra per-pid z-score

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
