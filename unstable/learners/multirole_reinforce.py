import math, ray, torch
from collections import defaultdict
from unstable.learners.multirole_base import MultiRoleBaseLearner
from unstable.learners.utils import role_adapter_name


@ray.remote
class MultiRoleREINFORCELearner(MultiRoleBaseLearner):
    def initialize_algorithm(self, max_train_len: int, max_generation_len: int, phase_loss_weights=None):
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        self.phase_loss_weights = dict(phase_loss_weights or {})
        if self.phase_loss_weights:
            total = sum(float(weight) for weight in self.phase_loss_weights.values())
            if total <= 0 or any(float(weight) <= 0 for weight in self.phase_loss_weights.values()):
                raise ValueError("phase_loss_weights must all be positive")
            self.phase_loss_weights = {
                phase: float(weight) / total for phase, weight in self.phase_loss_weights.items()
            }

    def _prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        return enc, advs, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, advs, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        out = self.policy_model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
        for i, o in enumerate(obs): mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {"loss": loss.item(), "logp_mean": seq_logp.mean().item(), "avg_train_len": avg_len, "pct_truncated": pct_truncated}

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
            phase_logp, phase_lengths, phase_truncated = [], [], []
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

            rewards = [float(step.reward) for step in phase_steps]
            prefix = f"phase/{phase}"
            metrics.update({
                f"{prefix}/samples": len(phase_steps),
                f"{prefix}/loss": phase_loss,
                f"{prefix}/logp_mean": self._mean(phase_logp),
                f"{prefix}/avg_train_len": self._mean(phase_lengths),
                f"{prefix}/pct_truncated": self._mean(phase_truncated),
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
