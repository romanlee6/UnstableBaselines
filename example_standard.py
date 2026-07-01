import time, ray, unstable
import unstable.reward_transformations as retra

MODEL_NAME = "Qwen/Qwen3-4B"
MAX_TRAIN_SEQ_LEN = 3000
MAX_GENERATION_LENGTH = 1024 

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}

# Ray init
ray.init(namespace="unstable")  

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"), # if num_players == num_actors, it's mirror self-play and no opponents will be sampled
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="IteratedPrisonersDilemma-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
        unstable.EvalEnvSpec(env_id="PublicGoodsGame-v0-train", num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-3.1-flash-lite-preview-20260303"),
])

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

# initialize model registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-3.1-flash-lite-preview-20260303"))

# initialize model sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

# Data Buffer
step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
    max_buffer_size=384*2, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# initialize the learner first so it reserves its GPU before the collector claims the rest for VLLM actors
learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(max_train_len=MAX_TRAIN_SEQ_LEN, max_generation_len=MAX_GENERATION_LENGTH))

# initialize the collector with 2 VLLM actors (1 learner GPU + 2 collector GPUs = 3 total)
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
    num_actors=2,
)


try:
    collector.collect.remote(num_train_workers=128, num_eval_workers=8)
    ray.get(learner.train.remote(100))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()