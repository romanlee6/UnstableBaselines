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
_MULTIROLE_EPISODE_ALGOS = {"a2c", "ppo", "grpo"}   # GAE / turn-level return-to-go -> per-role EpisodeBuffer
def _default_vllm_cfg(
    model_name: str,
    lora_cfg: dict,
    max_generation_len: int,
    max_train_len: Optional[int] = None,
    retain_recent_context: bool = False,
) -> dict:
    max_model_len = 8192
    max_prompt_tokens = max_model_len - max_generation_len
    if retain_recent_context and max_train_len is not None:
        max_prompt_tokens = min(max_prompt_tokens, max_train_len - max_generation_len)
    if max_prompt_tokens <= 0:
        raise ValueError("max_train_len/max_model_len must leave room for generated tokens")
    return {
        "model_name": model_name,
        "temperature": 0.6,
        "max_tokens": max_generation_len,
        "max_prompt_tokens": max_prompt_tokens,
        "prompt_prefix_tokens": 256,
        "max_parallel_seq": 128,
        "max_loras": 8,
        "lora_config": lora_cfg,
        "max_model_len": max_model_len,
    }

class _UBRun:
    def __init__(self, *, collector, learner): self.collector, self.learner = collector, learner
    def start(self, learning_steps: int = 200, num_collection_workers: int = 256, num_eval_workers: int = 16):
        try:
            self.collector.collect.remote(num_train_workers=num_collection_workers, num_eval_workers=num_eval_workers)
            ray.get(self.learner.train.remote(learning_steps))
        finally:
            ray.kill(self.collector, no_restart=True); ray.shutdown()

def build(*, model_name: str, train_envs: Sequence[unstable.TrainEnvSpec], eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None, env_sampling_strategy: str = "random", opponent_sampling_strategy: str = "none", fixed_opponents: Sequence[str] = ["google/gemini-2.0-flash-lite-001"], algorithm: str = "reinforce", max_train_len: Optional[int]=None, max_generation_len: int=4096, batch_size: int=384, mini_batch_size: int=1, learning_rate: float=1e-5, gradient_clipping: float=0.2, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool = False, buffer_size: Optional[int]=None, lora_config: Optional[dict]=None, vllm_config: Optional[dict]=None, wandb_project: str="UnstableBaselines", env_step_reward_scale: float=1.0):
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
    _step_xforms_single = [retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0), retra.EnvStepReward(env_step_reward_scale)]
    if algorithm in _STEP_BUFFER_ALGOS: buffer = unstable.StepBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms(_step_xforms_single), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    elif algorithm in _EPISODE_BUFFER_ALGOS: buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(max_buffer_size=buffer_size, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms(_step_xforms_single), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
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
    collector = unstable.Collector.options(name="Collector").remote(vllm_config=vllm_config or _default_vllm_cfg(model_name, _lora_cfg, max_generation_len, max_train_len), tracker=tracker, buffer=buffer, game_scheduler=game_scheduler)

    return _UBRun(collector=collector, learner=learner)


def build_multirole(*, model_name: str, role_pids: Sequence[int], train_envs: Sequence[unstable.TrainEnvSpec],
                    eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None,
                    eval_substitutions: Optional[dict]=None,
                    eval_provider: str="openrouter",
                    env_sampling_strategy: str = "random",
                    role_lora_cfgs: Optional[dict]=None,
                    shuffle_roles: bool = True,
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
                    initial_lora_paths: Optional[dict]=None,
                    wandb_project: str="UnstableBaselines",
                    run_name_suffix: Optional[str]=None,
                    env_step_reward_scale: float=1.0,
                    include_final_reward: bool=True,
                    include_invalid_move_reward: bool=True,
                    balanced_phases: Optional[Sequence[str]]=None,
                    phase_local_normalization: bool=False,
                    phase_loss_weights: Optional[dict]=None,
                    initial_step: int=1,
                    initial_samples_seen: Optional[dict]=None,
                    initial_training_state_path: Optional[str]=None,
                    max_oom_retries: int=3,
                    retain_recent_context: bool=False,
                    wandb_id: Optional[str]=None,
                    wandb_resume: Optional[str]=None,
                    run_name_override: Optional[str]=None):
    """Multi-role variant of build().

    - One shared learner on a single GPU holds N PEFT LoRAs (one per pid).
    - role_pids defines which pids are trainable. role_lora_cfgs is a Dict[int, dict] of
      LoraConfig kwargs per pid; if a pid is missing, _DEFAULT_LORA_CFG is broadcast.
    - eval_substitutions = {pid: model_name} swaps that pid's LoRA for an external
      model during eval games. eval_provider selects "openrouter" or "azure_ai".
    - algorithm: 'reinforce' | 'a2c' | 'ppo' | 'grpo'. 'a2c' and 'ppo' use per-role
      EpisodeBuffers (needed for GAE); 'reinforce' and 'grpo' use per-role StepBuffers.
    """
    assert algorithm in _MULTIROLE_ALGOS, f"algorithm='{algorithm}' not in {list(_MULTIROLE_ALGOS)}"
    if balanced_phases or phase_local_normalization or phase_loss_weights:
        assert algorithm == "reinforce", "phase-balanced sampling/loss is currently supported only for multirole REINFORCE"
    if not include_final_reward:
        assert algorithm == "reinforce", "disabling final reward is currently supported only for multirole REINFORCE"
    balanced_phases = tuple(balanced_phases or ())
    if balanced_phases:
        assert batch_size % len(balanced_phases) == 0, "batch_size must be divisible by len(balanced_phases)"
    if balanced_phases and phase_loss_weights:
        assert set(balanced_phases) == set(phase_loss_weights), "balanced_phases and phase_loss_weights must name the same phases"
    role_pids = list(role_pids)
    role_lora_cfgs = dict(role_lora_cfgs or {})
    for pid in role_pids: role_lora_cfgs.setdefault(pid, _DEFAULT_LORA_CFG)

    ray.init(namespace="unstable")

    assert env_sampling_strategy in _ENV_SAMPLERS, f"env_sampling_strategy='{env_sampling_strategy}' not found"
    env_sampler = _ENV_SAMPLERS[env_sampling_strategy](train_env_specs=train_envs, eval_env_specs=eval_envs)

    _suffix = f"-{run_name_suffix}" if run_name_suffix else ""
    _run_name = run_name_override or f"UB-multirole-{algorithm}-{model_name.split('/')[-1]}-{env_sampler.env_list()}{_suffix}-{int(time.time())}"
    tracker = unstable.Tracker.options(name="Tracker").remote(
        run_name=_run_name,
        wandb_project=wandb_project,
        wandb_id=wandb_id,
        wandb_resume=wandb_resume,
    )

    # one initial "base" entry per role pid (path=None means use the bare base model)
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    for pid in role_pids:
        ray.get(model_registry.add_checkpoint.remote(uid=f"base-role-{pid}", path=None, iteration=0, role_pid=pid))
    # On resume, promote the loaded LoRAs to the "current" ckpt per role so collectors serve
    # from them instead of the base model until step (initial_step) completes.
    if initial_lora_paths and initial_step > 1:
        _prev_iter = initial_step - 1
        for pid, path in initial_lora_paths.items():
            if path:
                ray.get(model_registry.add_checkpoint.remote(
                    uid=f"ckpt-role{pid}-{_prev_iter}", path=str(path), iteration=_prev_iter, role_pid=pid))

    # eval substitutions register their OpenRouter names as fixed entries (so update_ratings works)
    team_sampler = unstable.samplers.FixedRoleTeamSampler(
        model_registry=model_registry, role_pids=role_pids, eval_substitutions=eval_substitutions or {},
        eval_provider=eval_provider,
        shuffle_roles=shuffle_roles,
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

    final_xforms = [] if algorithm == "grpo" or not include_final_reward else [retra.RoleAdvantageByEnvFormatter()]
    step_xforms = [retra.RewardForFormat(1.5)]
    if include_invalid_move_reward:
        step_xforms.append(retra.PenaltyForInvalidMove(1.0, -1.0))
    step_xforms.append(retra.EnvStepReward(env_step_reward_scale))
    if env_step_reward_scale != 0.0 and not use_turn_scores:
        import warnings
        warnings.warn(
            "env_step_reward_scale is non-zero but use_turn_scores=False; "
            "per-step env rewards will be discarded by the learner. "
            "Set use_turn_scores=True to keep them.",
            RuntimeWarning, stacklevel=2,
        )
    # GRPO uses raw turn rewards to build return-to-go, then subtracts a per-(env, role, own_ckpt, opp_ckpts)
    # trajectory-mean baseline inside compute_advantages. Any batch-level reward whitening here would
    # distort R_{τ,k}; keep sampling_xforms empty for grpo.
    if algorithm == "grpo":
        sampling_xforms = []
    else:
        sampling_xforms = [retra.NormalizeRewardsByEnvPhase()] if phase_local_normalization else [retra.NormalizeRewardsByEnv(True)]
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
        if not use_episode:
            buffer_kwargs.update(
                role_pid=pid,
                include_final_reward=include_final_reward,
                balanced_phases=balanced_phases,
            )
        buffers[pid] = BufferCls.options(name=f"Buffer-role-{pid}").remote(**buffer_kwargs)

    LearnerCls = _MULTIROLE_ALGOS[algorithm]
    learner = LearnerCls.options(num_gpus=1, name="Learner").remote(
        model_name=model_name, role_lora_cfgs=role_lora_cfgs, batch_size=batch_size,
        mini_batch_size=mini_batch_size, learning_rate=learning_rate, grad_clip=gradient_clipping,
        buffers=buffers, tracker=tracker, model_registry=model_registry,
        activation_checkpointing=activation_checkpointing, gradient_checkpointing=gradient_checkpointing,
        use_trainer_cache=use_trainer_cache, initial_lora_paths=initial_lora_paths,
        initial_step=initial_step,
        initial_samples_seen=initial_samples_seen,
        initial_training_state_path=initial_training_state_path,
        max_oom_retries=max_oom_retries,
    )

    match algorithm:
        case "reinforce":
            ray.get(learner.initialize_algorithm.remote(
                max_train_len=max_train_len, max_generation_len=max_generation_len,
                phase_loss_weights=phase_loss_weights))
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
        vllm_config=vllm_config or _default_vllm_cfg(
            model_name, vllm_default_lora, max_generation_len, max_train_len,
            retain_recent_context=retain_recent_context,
        ),
        tracker=tracker, buffer=None, game_scheduler=game_scheduler, buffers=buffers,
    )

    return _UBRun(collector=collector, learner=learner)
