import math, ray, torch
from collections import defaultdict
from unstable.learners.multirole_base import MultiRoleBaseLearner
from unstable.learners.utils import role_adapter_name
from unstable.utils.context_window import completion_preserving_batch


def _validated_phase_loss_weights(phase_loss_weights=None, normalize=True):
    """Validate phase coefficients and optionally normalize them to sum to one."""
    weights = {
        phase: float(weight)
        for phase, weight in (phase_loss_weights or {}).items()
    }
    if not weights:
        return {}
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("phase_loss_weights must all be positive")
    if normalize:
        total = sum(weights.values())
        weights = {phase: weight / total for phase, weight in weights.items()}
    return weights


@ray.remote
class MultiRoleREINFORCELearner(MultiRoleBaseLearner):
    def initialize_algorithm(
        self,
        max_train_len: int,
        max_generation_len: int,
        phase_loss_weights=None,
        normalize_phase_loss_weights: bool = True,
    ):
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        self.phase_loss_weights = _validated_phase_loss_weights(
            phase_loss_weights,
            normalize=normalize_phase_loss_weights,
        )

    def _prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        prepared = completion_preserving_batch(
            self.tokenizer, obs, acts, self.max_train_len
        )
        avg_len = sum(prepared.original_lengths) / len(prepared.original_lengths)
        pct_truncated = (
            sum(dropped > 0 for dropped in prepared.prompt_tokens_dropped)
            / len(prepared.prompt_tokens_dropped)
        )
        avg_prompt_tokens_dropped = (
            sum(prepared.prompt_tokens_dropped) / len(prepared.prompt_tokens_dropped)
        )
        return (
            prepared.encoding.to(self.device), advs,
            prepared.prompt_lengths, prepared.sequence_lengths,
            avg_len, pct_truncated, avg_prompt_tokens_dropped,
        )

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        (
            enc, advs, prompt_lengths, sequence_lengths,
            avg_len, pct_truncated, avg_prompt_tokens_dropped,
        ) = self._prepare_batch(steps=steps)
        out = self.policy_model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = enc.attention_mask.bool()
        width = enc.input_ids.shape[1]
        for i, (prompt_len, sequence_len) in enumerate(
            zip(prompt_lengths, sequence_lengths)
        ):
            valid_start = 0 if self.tokenizer.padding_side == "right" else width - sequence_len
            mask[i, valid_start:valid_start + prompt_len] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {
            "loss": loss.item(),
            "logp_mean": seq_logp.mean().item(),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
            "avg_prompt_tokens_dropped": avg_prompt_tokens_dropped,
        }

    def _update(self, role_pid: int, batch):
        # activate this role's adapter so set_adapter routes grads to its lora_A/lora_B only.
        self.policy_model.set_adapter(role_adapter_name(role_pid))
        opt = self.policy_optimizers[role_pid]
        opt.zero_grad(set_to_none=True)

        if self.phase_loss_weights:
            return self._update_phase_balanced(role_pid, batch, opt)

        metrics_acc = {}
        for i in range(self.gradient_acc_steps):
            sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                update_metrics = self._mini_batch_update_step(sub, scaling=self.gradient_acc_steps)
            for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v / self.gradient_acc_steps
            self.logger.info(f"role-{role_pid} mini-step metrics: {update_metrics}")
        self.logger.info(f"role-{role_pid} step metrics: {metrics_acc}")
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], self.grad_clip)
        opt.step()
        return metrics_acc

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    def _update_phase_balanced(self, role_pid, batch, opt):
        grouped = defaultdict(list)
        for step in batch:
            grouped[step.phase].append(step)
        missing = set(self.phase_loss_weights) - set(grouped)
        if missing:
            raise ValueError(f"phase-balanced batch is missing phases: {sorted(missing)}")

        metrics = {}
        combined_loss = 0.0
        for phase, weight in self.phase_loss_weights.items():
            phase_steps = grouped[phase]
            num_chunks = int(math.ceil(len(phase_steps) / self.mini_batch_size))
            phase_loss = 0.0
            phase_logp, phase_lengths, phase_truncated, phase_tokens_dropped = [], [], [], []
            for start in range(0, len(phase_steps), self.mini_batch_size):
                sub = phase_steps[start:start + self.mini_batch_size]
                # Summing all micro-batch losses yields weight * mean phase loss.
                scaling = num_chunks / weight
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update = self._mini_batch_update_step(sub, scaling=scaling)
                phase_loss += update["loss"]
                phase_logp.append(update["logp_mean"])
                phase_lengths.append(update["avg_train_len"])
                phase_truncated.append(update["pct_truncated"])
                phase_tokens_dropped.append(update["avg_prompt_tokens_dropped"])

            rewards = [float(step.reward) for step in phase_steps]
            prefix = f"phase/{phase}"
            metrics.update({
                f"{prefix}/samples": len(phase_steps),
                f"{prefix}/loss": phase_loss,
                f"{prefix}/logp_mean": self._mean(phase_logp),
                f"{prefix}/avg_train_len": self._mean(phase_lengths),
                f"{prefix}/pct_truncated": self._mean(phase_truncated),
                f"{prefix}/avg_prompt_tokens_dropped": self._mean(phase_tokens_dropped),
                f"{prefix}/advantage_mean": self._mean(rewards),
                f"{prefix}/advantage_std": float(torch.tensor(rewards).std(unbiased=False).item()),
                f"{prefix}/advantage_min": min(rewards),
                f"{prefix}/advantage_max": max(rewards),
            })
            component_names = set().union(*(step.reward_components.keys() for step in phase_steps))
            for name in component_names:
                values = [float(step.reward_components.get(name, 0.0)) for step in phase_steps]
                metrics[f"{prefix}/reward/{name}"] = self._mean(values)
            combined_loss += phase_loss

        metrics["loss"] = combined_loss
        self.logger.info(f"role-{role_pid} phase-balanced step metrics: {metrics}")
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], self.grad_clip)
        opt.step()
        return metrics
