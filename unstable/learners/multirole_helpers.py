"""Shared helpers for multirole A2C / PPO / GRPO learners.

All helpers take the learner instance as their first arg so they can read
.policy_model, .tokenizer, .device, .max_train_len, .max_generation_len, etc.
without growing a new mixin or base class. They mirror the patterns established
by reinforce_learner.py and a2c_learner.py - in particular the Dr.GRPO
seq_logp = (tok_logp * mask).sum(1) / max_generation_len normalization and
the obs-only critic tokenization.
"""
import torch
from contextlib import nullcontext
from collections import defaultdict
from typing import List, Optional, Tuple, Callable, Dict

from unstable._types import Step
from unstable.learners.a2c_learner import compute_gae


def prepare_policy_batch(learner, steps: List[Step]):
    """Tokenize obs+act, pull advantages and returns from step.step_info.

    Returns (enc, advs, rets, obs_list, avg_len, pct_truncated). `rets` are NaN
    when no return was set (non-GAE modes); the caller decides whether to use them.
    """
    obs, acts, advs, rets = [], [], [], []
    for s in steps:
        info = s.step_info or {}
        obs.append(s.obs); acts.append(s.act)
        advs.append(float(info.get("advantage", s.reward)))
        rets.append(float(info.get("return", float("nan"))))
    advs_t = torch.tensor(advs, dtype=torch.float32, device=learner.device)
    rets_t = torch.tensor(rets, dtype=torch.float32, device=learner.device)
    combined = [o + a for o, a in zip(obs, acts)]
    lengths = [len(learner.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
    avg_len = sum(lengths) / len(lengths)
    pct_trunc = sum(l > learner.max_train_len for l in lengths) / len(lengths) if learner.max_train_len else 0
    enc = learner.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=learner.max_train_len).to(learner.device)
    return enc, advs_t, rets_t, obs, avg_len, pct_trunc


def prepare_state_batch(learner, steps: List[Step]):
    """Obs-only tokenization for critic inference (matches a2c_learner.py:47)."""
    obs = [s.obs for s in steps]
    return learner.tokenizer(obs, return_tensors="pt", padding=True, truncation=True, max_length=learner.max_train_len).to(learner.device)


def compute_seq_logp(learner, enc, obs_list: List[str]) -> torch.Tensor:
    """Dr.GRPO-style per-sequence log prob. Gradient-enabled forward by default."""
    out = learner.policy_model(**enc)
    logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
    tgt_ids = enc.input_ids[:, 1:]
    tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=learner.device)
    for i, o in enumerate(obs_list):
        mask[i, :len(learner.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
    mask = mask[:, 1:]
    return (tok_logp * mask).sum(1) / learner.max_generation_len


def snapshot_logp(learner, steps: List[Step], mini_batch_size: int, disable_adapter: bool = False) -> torch.Tensor:
    """No-grad chunked forward returning a flat [N] tensor on CPU.

    disable_adapter=True opens the PEFT model's disable_adapter() context so the
    forward runs through the bare base model - this is our reference policy for
    KL. The previously-active adapter is restored on context exit.

    NOTE: we deliberately do NOT call .eval()/.train() here. PEFT's PeftModel.train()
    can reset requires_grad on adapter params for multi-adapter models, breaking
    the subsequent gradient-enabled forward in the caller. Our LoRA dropout is 0,
    so skipping eval-mode has no downside.
    """
    parts = []
    ctx = learner.policy_model.disable_adapter() if disable_adapter else nullcontext()
    with torch.no_grad(), ctx:
        for i in range(0, len(steps), mini_batch_size):
            sub = steps[i:i + mini_batch_size]
            enc, _, _, obs, _, _ = prepare_policy_batch(learner, sub)
            with torch.autocast(device_type=learner.device.type, dtype=torch.bfloat16):
                parts.append(compute_seq_logp(learner, enc, obs).detach().float().cpu())
    return torch.cat(parts)


def compute_approx_kl_seq(seq_logp: torch.Tensor, ref_logp: torch.Tensor, kl_penalty: str = "k3") -> torch.Tensor:
    """Sequence-level KL estimators ported from MARSHAL functionals.py:164.

    All inputs are already /max_generation_len normalized. Returns per-sample
    KL contributions; caller averages over the batch.
    - k1: log_p - log_q  (biased, signed)
    - k2: 0.5 * (log_p - log_q)^2
    - k3: exp(log_q - log_p) - (log_q - log_p) - 1  (Schulman, non-negative)
    """
    diff = seq_logp - ref_logp
    if kl_penalty == "k1": return diff
    if kl_penalty == "k2": return 0.5 * diff.pow(2)
    if kl_penalty == "k3":
        kl = -diff
        return (kl.exp() - kl - 1).clamp(-10, 10)
    raise ValueError(f"unknown kl_penalty: {kl_penalty!r}; expected one of k1/k2/k3")


def resolve_turn_rewards(batch, use_turn_scores: bool) -> Tuple[List[Step], Optional[List[List[Step]]]]:
    """Resolve per-step rewards according to the turn-level toggle.

    Returns (flat_steps, episode_layout_or_None). When the input is a list of
    episodes (EpisodeBuffer), episode_layout is the original layout - needed by
    GAE. When the input is a flat list of steps (StepBuffer), episode_layout is None.

    use_turn_scores=True:  keep step.reward as-is (per-turn shaped reward, the
                            current default produced by the step_reward_transforms).
    use_turn_scores=False: collapse to episode-final reward only.
      - episodes input -> zero non-terminal step.reward, set terminal step.reward
        to step.step_info['env_reward'] (the final reward post-final_reward_transform).
      - flat input     -> overwrite every step.reward with its env_reward (each
        step in StepBuffer already has env_reward broadcast there - this drops the
        per-step format/penalty shaping deltas).
    """
    is_episodes = bool(batch) and isinstance(batch[0], list)
    if is_episodes:
        if not use_turn_scores:
            for ep in batch:
                if not ep: continue
                terminal_env_r = float((ep[-1].step_info or {}).get("env_reward", ep[-1].reward))
                for s in ep[:-1]: s.reward = 0.0
                ep[-1].reward = terminal_env_r
        flat = [s for ep in batch for s in ep]
        return flat, batch
    else:
        steps = list(batch)
        if not use_turn_scores:
            for s in steps:
                s.reward = float((s.step_info or {}).get("env_reward", s.reward))
        return steps, None


def _write_step_info(step: Step, **kv):
    step.step_info = {**(step.step_info or {}), **kv}


def compute_advantages(
    learner,
    steps: List[Step],
    episodes: Optional[List[List[Step]]],
    mode: str,
    critic: Optional[torch.nn.Module] = None,
    infer_mini_batch_size: int = 16,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> List[Step]:
    """Write step.step_info['advantage'] (and ['return'] for GAE) in-place. Returns the input list."""
    if mode == "reinforce":
        for s in steps:
            _write_step_info(s, advantage=float(s.reward), **{"return": float(s.reward)})
        return steps

    if mode == "grpo":
        # A_{τ,k} = R_{τ,k} - μ_{env, role, own_ckpt, opp_ckpts}
        # R_{τ,k}: turn-level return-to-go with γ=1.
        # μ: equal-trajectory-weighted mean of G_τ = R_{τ,0} across trajectories in the group.
        # Singleton-trajectory groups get zero advantage on every turn (no gradient contribution).
        assert episodes is not None, "grpo requires EpisodeBuffer input (List[List[Step]])"
        traj_r2g: List[List[float]] = []
        traj_returns: List[float] = []
        traj_keys: List[tuple] = []
        for ep in episodes:
            if not ep:
                traj_r2g.append([]); traj_returns.append(0.0); traj_keys.append(())
                continue
            r2g = [0.0] * len(ep)
            acc = 0.0
            for k in range(len(ep) - 1, -1, -1):
                acc += float(ep[k].reward)
                r2g[k] = acc
            traj_r2g.append(r2g)
            traj_returns.append(r2g[0])
            s0 = ep[0]
            traj_keys.append((s0.env_id, s0.role_pid, s0.own_model_uid, s0.opponent_model_uids))
        # equal-trajectory-weighted group means over G_τ
        group_returns: Dict[tuple, List[float]] = defaultdict(list)
        for key, G in zip(traj_keys, traj_returns):
            if key: group_returns[key].append(G)
        group_mu = {k: sum(v) / len(v) for k, v in group_returns.items()}
        group_size = {k: len(v) for k, v in group_returns.items()}
        for ep, r2g, key in zip(episodes, traj_r2g, traj_keys):
            if not ep: continue
            if group_size.get(key, 0) < 2:
                # singleton group: skip gradient contribution.
                for s, R in zip(ep, r2g):
                    _write_step_info(s, advantage=0.0, **{"return": float(R)})
                continue
            mu = group_mu[key]
            for s, R in zip(ep, r2g):
                _write_step_info(s, advantage=float(R - mu), **{"return": float(R)})
        return steps

    if mode == "gae":
        assert critic is not None and episodes is not None, "gae requires a critic and episode layout"
        all_values = []
        for i in range(0, len(steps), infer_mini_batch_size):
            sub = steps[i:i + infer_mini_batch_size]
            with torch.autocast(device_type=learner.device.type, dtype=torch.bfloat16), torch.no_grad():
                state_enc = prepare_state_batch(learner, sub)
                vals = critic(**state_enc)[:, 0]
            all_values.append(vals.float().cpu())
        all_values = torch.cat(all_values)
        ep_values = torch.split(all_values, [len(ep) for ep in episodes])
        for ep, vals in zip(episodes, ep_values):
            rewards = torch.tensor([s.reward for s in ep], dtype=torch.float32)
            adv = compute_gae(rewards, vals, gamma=gamma, gae_lambda=gae_lambda)
            rets = adv + vals
            for j, s in enumerate(ep):
                _write_step_info(s, advantage=float(adv[j]), **{"return": float(rets[j])})
        return steps

    raise ValueError(f"unknown adv_estimator: {mode!r}; expected one of 'gae', 'grpo', 'reinforce'")
