import ray, time
from typing import List, Sequence, Optional

import unstable
import unstable.reward_transformations as retra


_DEFAULT_LORA_CFG = {"lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]}
_ENV_SAMPLERS = {"random": unstable.samplers.env_samplers.UniformRandomEnvSampler}
_OPP_SAMPLERS = {"none": unstable.samplers.model_samplers.BaseModelSampler, "mirror": unstable.samplers.model_samplers.BaseModelSampler, "fixed": unstable.samplers.model_samplers.FixedOpponentModelSampler}
_STEP_BUFFER_ALGOS = ["reinforce"]
_EPISODE_BUFFER_ALGOS = ["a2c"]
_ALGOS = {"reinforce": unstable.REINFORCELearner, "a2c": unstable.A2CLearner}
_MULTIROLE_ALGOS = {
    "reinforce": unstable.MultiRoleREINFORCELearner,
    "a2c":       unstable.MultiRoleA2CLearner,
    "ppo":       unstable.MultiRolePPOLearner,   # adv_estimator='gae'
    "grpo":      unstable.MultiRolePPOLearner,   # adv_estimator='grpo'
}
_MULTIROLE_EPISODE_ALGOS = {"a2c", "ppo"}   # GAE -> per-role EpisodeBuffer
def _default_vllm_cfg(model_name: str, lora_cfg: dict, max_generation_len: int) -> dict: return {"model_name": model_name, "temperature": 0.6, "max_tokens": max_generation_len, "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_cfg, "max_model_len": 8192}

class _UBRun:
    def __init__(self, *, collector, learner): self.collector, self.learner = collector, learner
    def start(self, learning_steps: int = 200, num_collection_workers: int = 256, num_eval_workers: int = 16):
        try:
            self.collector.collect.remote(num_train_workers=num_collection_workers, num_eval_workers=num_eval_workers)
            ray.get(self.learner.train.remote(learning_steps))
        finally:
            ray.kill(self.collector, no_restart=True); ray.shutdown()

def build(*, model_name: str, train_envs: Sequence[unstable.TrainEnvSpec], eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None, env_sampling_strategy: str = "random", opponent_sampling_strategy: str = "none", fixed_opponents: Sequence[str] = ["google/gemini-2.0-flash-lite-001"], algorithm: str = "reinforce", max_train_len: Optional[int]=None, max_generation_len: int=4096, batch_size: int=384, mini_batch_size: int=1, learning_rate: float=1e-5, gradient_clipping: float=0.2, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool = False, buffer_size: Optional[int]=None, lora_config: Optional[dict]=None, vllm_config: Optional[dict]=None, wandb_project: str="UnstableBaselines"):
    # Ray init
    ray.init(namespace="unstable")  
    
    # env sampler
    assert env_sampling_strategy in _ENV_SAMPLERS, f"env_sampling_strategy='{env_sampling_strategy}' not found. Please use one of: {list(_ENV_SAMPLERS.keys())}"
    env_sampler = _ENV_SAMPLERS[env_sampling_strategy](train_env_specs=train_envs, eval_env_specs=eval_envs)

    # tracker
    tracker = unstable.Tracker.options(name="Tracker").remote(run_name=f"UnstableBaselines-{model_name.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", wandb_project="UnstableBaselines") 

    # initialize model registry
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
    for f_opp_name in fixed_opponents: ray.get(model_registry.add_fixed.remote(name=f_opp_name))

    # initialize opponent sampler
    assert opponent_sampling_strategy in _OPP_SAMPLERS, f"opponent_sampling_strategy='{opponent_sampling_strategy}' not found. Please use one of: {list(_OPP_SAMPLERS.keys())}"
    model_sampler = _OPP_SAMPLERS[opponent_sampling_strategy](model_registry=model_registry)

    # build game scheduler
    game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

    # build buffer TODO maybe move the reward transformations outside
    buffer_size = buffer_size or batch_size*2
    if algorithm in _STEP_BUFFER_ALGOS: buffer = unstable.StepBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    elif algorithm in _EPISODE_BUFFER_ALGOS: buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    else: raise NotImplementedError(f"The algorithm used ({algorithm}) has not been allocated to a specific buffer type.")

    # initialize the learner first so it reserves its GPU before the collector claims the rest for VLLM actors
    _lora_cfg = lora_config or _DEFAULT_LORA_CFG
    assert algorithm in _ALGOS, f"algorithm='{algorithm}' not found. Please use one of: {list(_ALGOS.keys())}"
    learner = _ALGOS[algorithm].options(num_gpus=1, name="Learner").remote(model_name=model_name, lora_cfg=_lora_cfg, batch_size=batch_size, mini_batch_size=mini_batch_size, learning_rate=learning_rate, grad_clip=gradient_clipping, buffer=buffer, tracker=tracker, model_registry=model_registry, activation_checkpointing=activation_checkpointing, gradient_checkpointing=gradient_checkpointing, use_trainer_cache=use_trainer_cache)
    match algorithm:
        case "reinforce":   ray.get(learner.initialize_algorithm.remote(max_train_len=max_train_len, max_generation_len=max_generation_len))
        case "a2c":         ray.get(learner.initialize_algorithm.remote(infer_mini_batch_size=16, critic_learning_rate=5e-5, normalize_adv=True, max_train_len=max_train_len, max_generation_len=max_generation_len)) # TODO find better solution
        case _:             ray.get(learner.initialize_algorithm.remote())

    # initialize the collector after the learner so it only claims the remaining GPUs for VLLM actors
    collector = unstable.Collector.options(name="Collector").remote(vllm_config=vllm_config or _default_vllm_cfg(model_name, _lora_cfg, max_generation_len), tracker=tracker, buffer=buffer, game_scheduler=game_scheduler)

    return _UBRun(collector=collector, learner=learner)


def build_multirole(*, model_name: str, role_pids: Sequence[int], train_envs: Sequence[unstable.TrainEnvSpec],
                    eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None,
                    eval_substitutions: Optional[dict]=None,
                    env_sampling_strategy: str = "random",
                    role_lora_cfgs: Optional[dict]=None,
                    algorithm: str = "reinforce",
                    adv_estimator: Optional[str]=None,
                    use_turn_scores: bool = True,
                    clip_eps: float = 0.2,
                    n_epochs: int = 2,
                    critic_learning_rate: float = 5e-5,
                    gamma: float = 1.0, gae_lambda: float = 1.0,
                    kl_loss_coef: float = 0.0, kl_penalty: str = "k3",
                    normalize_adv_by_pid: bool = False,
                    max_train_len: Optional[int]=None, max_generation_len: int=4096,
                    batch_size: int=384, mini_batch_size: int=1, learning_rate: float=1e-5,
                    gradient_clipping: float=0.2, activation_checkpointing: bool=True,
                    gradient_checkpointing: bool=True, use_trainer_cache: bool=False,
                    buffer_size: Optional[int]=None, vllm_config: Optional[dict]=None,
                    wandb_project: str="UnstableBaselines"):
    """Multi-role variant of build().

    - One shared learner on a single GPU holds N PEFT LoRAs (one per pid).
    - role_pids defines which pids are trainable. role_lora_cfgs is a Dict[int, dict] of
      LoraConfig kwargs per pid; if a pid is missing, _DEFAULT_LORA_CFG is broadcast.
    - eval_substitutions = {pid: openrouter_name} swaps that pid's LoRA for an external
      model during eval games (e.g. {1: "google/gemini-3.1-flash-lite-preview"}).
    - algorithm: 'reinforce' | 'a2c' | 'ppo' | 'grpo'. 'a2c' and 'ppo' use per-role
      EpisodeBuffers (needed for GAE); 'reinforce' and 'grpo' use per-role StepBuffers.
    """
    assert algorithm in _MULTIROLE_ALGOS, f"algorithm='{algorithm}' not in {list(_MULTIROLE_ALGOS)}"
    role_pids = list(role_pids)
    role_lora_cfgs = dict(role_lora_cfgs or {})
    for pid in role_pids: role_lora_cfgs.setdefault(pid, _DEFAULT_LORA_CFG)

    ray.init(namespace="unstable")

    assert env_sampling_strategy in _ENV_SAMPLERS, f"env_sampling_strategy='{env_sampling_strategy}' not found"
    env_sampler = _ENV_SAMPLERS[env_sampling_strategy](train_env_specs=train_envs, eval_env_specs=eval_envs)

    tracker = unstable.Tracker.options(name="Tracker").remote(
        run_name=f"UB-multirole-{algorithm}-{model_name.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}",
        wandb_project=wandb_project,
    )

    # one initial "base" entry per role pid (path=None means use the bare base model)
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    for pid in role_pids:
        ray.get(model_registry.add_checkpoint.remote(uid=f"base-role-{pid}", path=None, iteration=0, role_pid=pid))

    # eval substitutions register their OpenRouter names as fixed entries (so update_ratings works)
    team_sampler = unstable.samplers.FixedRoleTeamSampler(
        model_registry=model_registry, role_pids=role_pids, eval_substitutions=eval_substitutions or {},
    )

    game_scheduler = unstable.game_scheduler.MultiRoleGameScheduler.options(name="GameScheduler").remote(
        team_sampler=team_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()),
    )

    # Per-role buffers. GRPO subtracts its own group baseline at update time, so we
    # drop RoleAdvantageByEnvFormatter to avoid double-subtraction. Per-pid advantage
    # normalization is opt-in via normalize_adv_by_pid.
    buffer_size = buffer_size or batch_size * 2
    use_episode = algorithm in _MULTIROLE_EPISODE_ALGOS
    BufferCls = unstable.EpisodeBuffer if use_episode else unstable.StepBuffer

    final_xforms = [] if algorithm == "grpo" else [retra.RoleAdvantageByEnvFormatter()]
    step_xforms = [retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]
    sampling_xforms = [retra.NormalizeRewardsByEnv(True)]
    if normalize_adv_by_pid:
        sampling_xforms.append(retra.NormalizeAdvantagesByPidEnv(z_score=True))

    buffers = {}
    for pid in role_pids:
        # one buffer per role; each gets its own reward-transformation pipeline so the
        # per-(env, pid) baseline in RoleAdvantageByEnvFormatter doesn't bleed across roles.
        buffer_kwargs = dict(
            max_buffer_size=buffer_size, tracker=tracker,
            final_reward_transformation=retra.ComposeFinalRewardTransforms(final_xforms),
            step_reward_transformation=retra.ComposeStepRewardTransforms(step_xforms),
            sampling_reward_transformation=retra.ComposeSamplingRewardTransforms(sampling_xforms),
        )
        # StepBuffer accepts role_pid for per-role log-file suffixing; EpisodeBuffer does not.
        if not use_episode: buffer_kwargs["role_pid"] = pid
        buffers[pid] = BufferCls.options(name=f"Buffer-role-{pid}").remote(**buffer_kwargs)

    LearnerCls = _MULTIROLE_ALGOS[algorithm]
    learner = LearnerCls.options(num_gpus=1, name="Learner").remote(
        model_name=model_name, role_lora_cfgs=role_lora_cfgs, batch_size=batch_size,
        mini_batch_size=mini_batch_size, learning_rate=learning_rate, grad_clip=gradient_clipping,
        buffers=buffers, tracker=tracker, model_registry=model_registry,
        activation_checkpointing=activation_checkpointing, gradient_checkpointing=gradient_checkpointing,
        use_trainer_cache=use_trainer_cache,
    )

    match algorithm:
        case "reinforce":
            ray.get(learner.initialize_algorithm.remote(
                max_train_len=max_train_len, max_generation_len=max_generation_len))
        case "a2c":
            ray.get(learner.initialize_algorithm.remote(
                max_train_len=max_train_len, max_generation_len=max_generation_len,
                use_turn_scores=use_turn_scores,
                critic_learning_rate=critic_learning_rate,
                gamma=gamma, gae_lambda=gae_lambda,
                kl_loss_coef=kl_loss_coef, kl_penalty=kl_penalty,
                normalize_adv=False))
        case "ppo":
            ray.get(learner.initialize_algorithm.remote(
                max_train_len=max_train_len, max_generation_len=max_generation_len,
                adv_estimator=adv_estimator or "gae",
                use_turn_scores=use_turn_scores,
                clip_eps=clip_eps, n_epochs=n_epochs,
                critic_learning_rate=critic_learning_rate,
                gamma=gamma, gae_lambda=gae_lambda,
                kl_loss_coef=kl_loss_coef, kl_penalty=kl_penalty,
                normalize_adv=True))
        case "grpo":
            ray.get(learner.initialize_algorithm.remote(
                max_train_len=max_train_len, max_generation_len=max_generation_len,
                adv_estimator=adv_estimator or "grpo",
                use_turn_scores=use_turn_scores,
                clip_eps=clip_eps, n_epochs=n_epochs,
                kl_loss_coef=kl_loss_coef, kl_penalty=kl_penalty,
                normalize_adv=False))

    # use the first role's lora_cfg for the vllm default; vllm supports multiple loras via LoRARequest per inference
    vllm_default_lora = role_lora_cfgs[role_pids[0]]
    collector = unstable.Collector.options(name="Collector").remote(
        vllm_config=vllm_config or _default_vllm_cfg(model_name, vllm_default_lora, max_generation_len),
        tracker=tracker, buffer=None, game_scheduler=game_scheduler, buffers=buffers,
    )

    return _UBRun(collector=collector, learner=learner)
