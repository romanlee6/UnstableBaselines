"""Multirole PPO / GRPO learner.

One class handles both algorithms because they share the clipped surrogate
objective; only the advantage estimator differs. `adv_estimator` selects:
- "gae":       per-role critic + GAE bootstrap (vanilla PPO)
- "grpo":      no critic, group-baseline-subtracted advantages (vanilla GRPO)
- "reinforce": no critic, raw step.reward as advantage (REINFORCE with PPO clip)

Toggles: use_turn_scores (per-turn vs sequence-level rewards), normalize_adv
(batch-level z-score), kl_loss_coef (KL against the pure base model via PEFT
disable_adapter()).
"""
import ray, random, torch
from typing import Dict, Optional

from unstable.learners.multirole_base import MultiRoleBaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt, role_adapter_name
from unstable.learners.multirole_helpers import (
    prepare_policy_batch, compute_seq_logp, snapshot_logp,
    compute_approx_kl_seq, resolve_turn_rewards, compute_advantages,
    prepare_state_batch,
)


@ray.remote
class MultiRolePPOLearner(MultiRoleBaseLearner):
    def initialize_algorithm(
        self,
        max_train_len: int,
        max_generation_len: int,
        adv_estimator: str = "gae",          # "gae" | "grpo" | "reinforce"
        use_turn_scores: bool = True,
        clip_eps: float = 0.2,
        n_epochs: int = 2,
        critic_learning_rate: float = 5e-5,
        infer_mini_batch_size: int = 16,
        normalize_adv: bool = False,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        kl_loss_coef: float = 0.0,
        kl_penalty: str = "k3",
        initial_critic_lora_paths: Optional[Dict[int, str]] = None,
    ):
        assert adv_estimator in ("gae", "grpo", "reinforce"), f"bad adv_estimator: {adv_estimator!r}"
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        self.adv_estimator = adv_estimator
        self.use_turn_scores = use_turn_scores
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.infer_mini_batch_size = infer_mini_batch_size
        self.normalize_adv = normalize_adv
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.kl_loss_coef, self.kl_penalty = kl_loss_coef, kl_penalty
        self._use_critic = (adv_estimator == "gae")

        self.critics: Dict[int, torch.nn.Module] = {}
        self.critic_optimizers: Dict[int, torch.optim.Optimizer] = {}
        if self._use_critic:
            for pid in self.role_pids:
                cfg = self.role_lora_cfgs[pid]
                init_path = (initial_critic_lora_paths or {}).get(pid)
                critic, _ = build_peft_model(self.model_name, self.device, cfg, init_path, critic_model=True)
                if self.gradient_checkpointing: critic.gradient_checkpointing_enable()
                if self.activation_checkpointing: enable_full_activation_ckpt(critic)
                self.critics[pid] = critic
                self.critic_optimizers[pid] = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, critic.parameters()),
                    lr=critic_learning_rate,
                )
                self.logger.info(f"built critic for role-{pid}")

    def _update(self, role_pid: int, batch):
        self.policy_model.set_adapter(role_adapter_name(role_pid))
        policy_opt = self.policy_optimizers[role_pid]
        critic = self.critics.get(role_pid) if self._use_critic else None
        critic_opt = self.critic_optimizers.get(role_pid) if self._use_critic else None

        flat_steps, episodes = resolve_turn_rewards(batch, self.use_turn_scores)
        if self.adv_estimator == "gae":
            assert episodes is not None, "adv_estimator='gae' requires EpisodeBuffer (List[List[Step]])"

        train_steps = compute_advantages(
            self, flat_steps, episodes, mode=self.adv_estimator, critic=critic,
            infer_mini_batch_size=self.infer_mini_batch_size,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        if self.normalize_adv:
            advs = torch.tensor([(s.step_info or {})["advantage"] for s in train_steps])
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for s, a in zip(train_steps, advs):
                s.step_info = {**(s.step_info or {}), "advantage": float(a)}

        if len(train_steps) > self.batch_size:
            train_steps = random.sample(train_steps, self.batch_size)
        assert len(train_steps) >= self.batch_size, f"need {self.batch_size} steps, got {len(train_steps)}"

        # Snapshot old_logp once - frozen across the n_epochs loop. ref_logp only
        # if KL is on. Order matters: ref snapshot opens disable_adapter() which
        # would also disable the role's adapter, so we run policy old_logp first
        # while the role adapter is still active, then the ref pass.
        old_logp_all = snapshot_logp(self, train_steps, self.mini_batch_size, disable_adapter=False)
        ref_logp_all = None
        if self.kl_loss_coef > 0:
            ref_logp_all = snapshot_logp(self, train_steps, self.mini_batch_size, disable_adapter=True)
        # Defensive: re-activate this role's adapter before the gradient loop.
        # PEFT's disable_adapter() context restores state on exit, but the
        # snapshot pass above sometimes leaves requires_grad=False on the active
        # adapter (multi-adapter quirk). set_adapter resets requires_grad correctly.
        self.policy_model.set_adapter(role_adapter_name(role_pid))

        metrics_acc: Dict[str, float] = {}
        scaling = float(self.gradient_acc_steps)
        for epoch in range(self.n_epochs):
            policy_opt.zero_grad(set_to_none=True)
            if critic_opt is not None: critic_opt.zero_grad(set_to_none=True)
            for i in range(self.gradient_acc_steps):
                sl = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
                sub = train_steps[sl]
                old_chunk = old_logp_all[sl]
                ref_chunk = ref_logp_all[sl] if ref_logp_all is not None else None
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    m = self._mini_step(sub, scaling, old_chunk, ref_chunk, critic)
                denom = scaling * float(self.n_epochs)
                for k, v in m.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v / denom
                self.logger.info(f"role-{role_pid} epoch {epoch} mini-step: {m}")

            torch.nn.utils.clip_grad_norm_(policy_opt.param_groups[0]['params'], self.grad_clip)
            if critic is not None:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
            policy_opt.step()
            if critic_opt is not None: critic_opt.step()

        self.logger.info(f"role-{role_pid} step metrics: {metrics_acc}")
        return metrics_acc

    def _mini_step(self, steps, scaling: float, old_logp_chunk: torch.Tensor,
                   ref_logp_chunk: Optional[torch.Tensor], critic):
        enc, advs, rets, obs, avg_len, pct_trunc = prepare_policy_batch(self, steps)
        seq_logp = compute_seq_logp(self, enc, obs)

        old_chunk = old_logp_chunk.to(self.device, dtype=seq_logp.dtype)
        ratio = torch.exp(seq_logp - old_chunk)
        unclipped = ratio * advs
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        pg_loss = -torch.min(unclipped, clipped).mean() / scaling

        total = pg_loss
        kl_metric = None
        if ref_logp_chunk is not None:
            ref_chunk = ref_logp_chunk.to(self.device, dtype=seq_logp.dtype)
            kl = compute_approx_kl_seq(seq_logp, ref_chunk, self.kl_penalty)
            kl_loss = kl.mean() / scaling
            total = total + self.kl_loss_coef * kl_loss
            kl_metric = kl_loss.item()
        total.backward()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - (seq_logp - old_chunk)).mean().item()
            clipfrac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

        out = {
            "pg_loss": pg_loss.item(), "approx_kl": approx_kl, "clipfrac": clipfrac,
            "logp_mean": seq_logp.mean().item(), "ratio_mean": ratio.mean().item(),
            "avg_train_len": avg_len, "pct_truncated": pct_trunc,
        }
        if kl_metric is not None: out["kl_loss"] = kl_metric

        if critic is not None:
            state_enc = prepare_state_batch(self, steps)
            value_pred = critic(**state_enc)[:, 0]
            v_loss = 0.5 * ((value_pred - rets) ** 2).mean() / scaling
            v_loss.backward()
            out["value_loss"] = v_loss.item()
            out["value_mae"] = (value_pred - rets).abs().mean().item()
        torch.cuda.empty_cache()
        return out
