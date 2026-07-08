"""Multirole A2C learner.

Per-role critics + per-role policy LoRAs on a shared base model. Each role's
update step is a single epoch of policy + critic backprop with GAE-computed
advantages. Optional KL penalty against the pure base model (via PEFT
disable_adapter()) and the use_turn_scores toggle (zero non-terminal rewards
when off).

This is the multirole analogue of A2CLearner (a2c_learner.py).
"""
import ray, torch
from typing import Dict, Optional

from unstable.learners.multirole_base import MultiRoleBaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt, role_adapter_name
from unstable.learners.multirole_helpers import (
    prepare_policy_batch, compute_seq_logp, snapshot_logp,
    compute_approx_kl_seq, resolve_turn_rewards, compute_advantages,
    prepare_state_batch,
)


@ray.remote
class MultiRoleA2CLearner(MultiRoleBaseLearner):
    def initialize_algorithm(
        self,
        max_train_len: int,
        max_generation_len: int,
        use_turn_scores: bool = True,
        critic_learning_rate: float = 5e-5,
        infer_mini_batch_size: int = 16,
        normalize_adv: bool = False,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        kl_loss_coef: float = 0.0,
        kl_penalty: str = "k3",
        initial_critic_lora_paths: Optional[Dict[int, str]] = None,
    ):
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        self.use_turn_scores = use_turn_scores
        self.infer_mini_batch_size = infer_mini_batch_size
        self.normalize_adv = normalize_adv
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.kl_loss_coef, self.kl_penalty = kl_loss_coef, kl_penalty

        # one critic per role - mirrors A2CLearner.initialize_algorithm (a2c_learner.py:32)
        self.critics: Dict[int, torch.nn.Module] = {}
        self.critic_optimizers: Dict[int, torch.optim.Optimizer] = {}
        for pid in self.role_pids:
            cfg = self.role_lora_cfgs[pid]
            init_path = (initial_critic_lora_paths or {}).get(pid)
            critic, _ = build_peft_model(self.model_name, self.device, cfg, init_path, critic_model=True)
            if self.gradient_checkpointing: critic.enable_input_require_grads(); critic.gradient_checkpointing_enable()
            if self.activation_checkpointing: enable_full_activation_ckpt(critic)
            self.critics[pid] = critic
            self.critic_optimizers[pid] = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, critic.parameters()),
                lr=critic_learning_rate,
            )
            self.logger.info(f"built critic for role-{pid}")

    def _update(self, role_pid: int, batch):
        self.policy_model.set_adapter(role_adapter_name(role_pid))
        critic = self.critics[role_pid]
        policy_opt = self.policy_optimizers[role_pid]
        critic_opt = self.critic_optimizers[role_pid]

        # 1) Resolve per-turn vs sequence-level rewards. A2C needs EpisodeBuffer
        #    output (List[List[Step]]).
        flat_steps, episodes = resolve_turn_rewards(batch, self.use_turn_scores)
        assert episodes is not None, "MultiRoleA2CLearner requires EpisodeBuffer (List[List[Step]])"

        # 2) Compute GAE advantages + returns using this role's critic.
        train_steps = compute_advantages(
            self, flat_steps, episodes, mode="gae", critic=critic,
            infer_mini_batch_size=self.infer_mini_batch_size,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        # 3) Optional batch-level z-score on the advantage signal.
        if self.normalize_adv:
            advs = torch.tensor([(s.step_info or {})["advantage"] for s in train_steps])
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for s, a in zip(train_steps, advs):
                s.step_info = {**(s.step_info or {}), "advantage": float(a)}

        # 4) Subsample to exactly batch_size (matches A2CLearner._update flow at a2c_learner.py:114).
        import random
        if len(train_steps) > self.batch_size:
            train_steps = random.sample(train_steps, self.batch_size)
        assert len(train_steps) >= self.batch_size, f"need {self.batch_size} steps, got {len(train_steps)}"

        # 5) Optional reference-policy snapshot for KL (only if enabled).
        ref_logp_all = None
        if self.kl_loss_coef > 0:
            ref_logp_all = snapshot_logp(self, train_steps, self.mini_batch_size, disable_adapter=True)

        # 6) Single-epoch policy+critic update with gradient accumulation.
        policy_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        metrics_acc = {}
        scaling = float(self.gradient_acc_steps)
        for i in range(self.gradient_acc_steps):
            sl = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
            sub = train_steps[sl]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                m = self._mini_step(sub, scaling, ref_logp_all[sl] if ref_logp_all is not None else None, critic)
            for k, v in m.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v / scaling
            self.logger.info(f"role-{role_pid} mini-step metrics: {m}")

        torch.nn.utils.clip_grad_norm_(policy_opt.param_groups[0]['params'], self.grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        policy_opt.step()
        critic_opt.step()
        self.logger.info(f"role-{role_pid} step metrics: {metrics_acc}")
        return metrics_acc

    def _mini_step(self, steps, scaling: float, ref_logp_chunk: Optional[torch.Tensor], critic):
        enc, advs, rets, obs, avg_len, pct_trunc = prepare_policy_batch(self, steps)
        seq_logp = compute_seq_logp(self, enc, obs)
        pg = -(advs * seq_logp).mean() / scaling

        total = pg
        kl_metric = None
        if ref_logp_chunk is not None:
            ref_chunk = ref_logp_chunk.to(self.device, dtype=seq_logp.dtype)
            kl = compute_approx_kl_seq(seq_logp, ref_chunk, self.kl_penalty)
            kl_loss = kl.mean() / scaling
            total = total + self.kl_loss_coef * kl_loss
            kl_metric = kl_loss.item()
        total.backward()

        state_enc = prepare_state_batch(self, steps)
        value_pred = critic(**state_enc)[:, 0]
        v_loss = 0.5 * ((value_pred - rets) ** 2).mean() / scaling
        v_loss.backward()
        torch.cuda.empty_cache()

        out = {
            "policy_loss": pg.item(), "value_loss": v_loss.item(),
            "logp_mean": seq_logp.mean().item(),
            "value_mae": (value_pred - rets).abs().mean().item(),
            "avg_train_len": avg_len, "pct_truncated": pct_trunc,
        }
        if kl_metric is not None: out["kl_loss"] = kl_metric
        return out
