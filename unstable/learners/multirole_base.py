import os, ray, torch, time, pathlib
from typing import List, Dict, Any, Optional

from unstable.buffers import BaseBuffer
from unstable.trackers import BaseTracker
from unstable.learners.utils import build_multi_adapter_peft_model, enable_full_activation_ckpt, role_adapter_name
from unstable.utils import setup_logger


class MultiRoleBaseLearner:
    """Single Ray learner process that holds N PEFT LoRA adapters on one shared base model.

    Each role/pid has: one adapter, one AdamW, one StepBuffer (Ray actor handle).
    `_update(role_pid, batch)` is implemented by algorithm subclasses (e.g. MultiRoleREINFORCELearner).
    Iteration loop polls per-role buffers and updates whichever roles have a fresh batch
    (liveness over barrier).
    """

    def __init__(self, model_name: str, role_lora_cfgs: Dict[int, Dict[str, Any]],
                 batch_size: int, mini_batch_size: int, learning_rate: float, grad_clip: float,
                 buffers: Dict[int, BaseBuffer], tracker: BaseTracker, model_registry,
                 activation_checkpointing: bool=True, gradient_checkpointing: bool=True,
                 use_trainer_cache: bool=False, initial_lora_paths: Optional[Dict[int, str]]=None,
                 initial_step: int=1, initial_samples_seen: Optional[Dict[int, int]]=None,
                 initial_training_state_path: Optional[str]=None):
        self.model_name, self.role_lora_cfgs = model_name, role_lora_cfgs
        self.buffers, self.tracker, self.model_registry = buffers, tracker, model_registry
        self.logger = setup_logger("multirole_learner", ray.get(tracker.get_log_dir.remote()))
        self.use_trainer_cache, self.gradient_checkpointing, self.activation_checkpointing = use_trainer_cache, gradient_checkpointing, activation_checkpointing
        self.batch_size, self.mini_batch_size, self.lr, self.grad_clip = batch_size, mini_batch_size, learning_rate, grad_clip
        self.gradient_acc_steps = self.batch_size // self.mini_batch_size
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote())); self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.bfloat16)

        gpu_ids = ray.get_gpu_ids()
        self.device = (torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu"))
        self.policy_model, self.tokenizer, self.role_pids = build_multi_adapter_peft_model(
            model_name, self.device, role_lora_cfgs, initial_lora_paths,
        )
        self.policy_model.to(torch.bfloat16)

        if not self.use_trainer_cache:    self.policy_model.config.use_cache = False
        if self.gradient_checkpointing:   self.policy_model.enable_input_require_grads(); self.policy_model.gradient_checkpointing_enable()
        if self.activation_checkpointing: enable_full_activation_ckpt(self.policy_model)

        # one AdamW per role. set_adapter(role) flips requires_grad so only that role's params show up.
        # filter by dotted form f".{adapter_name}." to avoid role-1 substring-matching role-10.
        self.policy_optimizers: Dict[int, torch.optim.Optimizer] = {}
        for pid in self.role_pids:
            adapter_name = role_adapter_name(pid)
            self.policy_model.set_adapter(adapter_name)
            params = [p for n, p in self.policy_model.named_parameters()
                      if p.requires_grad and f".{adapter_name}." in n]
            assert params, f"no trainable params found for adapter {adapter_name}; check PEFT wiring"
            self.policy_optimizers[pid] = torch.optim.AdamW(params, lr=learning_rate)
            self.logger.info(f"built AdamW for role-{pid} over {len(params)} param tensors")

        self._step = initial_step
        initial_samples_seen = initial_samples_seen or {}
        self._samples_seen: Dict[int, int] = {
            pid: int(initial_samples_seen.get(pid, 0)) for pid in self.role_pids
        }
        # Algorithm-specific initialization happens after __init__, so defer the
        # restore until train() starts.  This also makes the hook safe for future
        # optimizers created by initialize_algorithm().
        self._initial_training_state_path = initial_training_state_path
        self._training_state_restored = False

    def initialize_algorithm(self, *args, **kwargs): raise NotImplementedError
    def _update(self, role_pid: int, batch):         raise NotImplementedError

    def _ready_roles(self) -> List[int]:
        ready = []
        for pid in self.role_pids:
            if ray.get(self.buffers[pid].ready_for_batch.remote(self.batch_size)):
                ready.append(pid)
        return ready

    def train(self, iterations: int):
        self._restore_training_state()
        self.logger.info(f"Starting multi-role training loop over roles={self.role_pids}")
        while self._step < iterations:
            try:
                while not self._ready_roles(): time.sleep(0.2)
                ready = self._ready_roles()
                self.logger.info(f"step {self._step}: ready roles = {ready}")
                for pid in ready:
                    batch: List = ray.get(self.buffers[pid].get_batch.remote(self.batch_size))
                    self._samples_seen[pid] += self.batch_size
                    metrics = self._update(role_pid=pid, batch=batch)

                    opt = self.policy_optimizers[pid]
                    grad_norm = sum(p.grad.data.norm(2).item()**2 for p in opt.param_groups[0]['params'] if p.grad is not None) ** 0.5
                    log = {f"role-{pid}/{k}": v for k, v in metrics.items()}
                    log.update({"step": self._step, f"role-{pid}/samples_seen": self._samples_seen[pid], f"role-{pid}/lr": opt.param_groups[0]["lr"], f"role-{pid}/grad_norm": grad_norm})
                    self.tracker.log_learner.remote(log)

                    ckpt_path = self._save_checkpoint(pid)
                    try:
                        self.model_registry.add_checkpoint.remote(uid=f"ckpt-role{pid}-{self._step}", path=str(ckpt_path), iteration=self._step, role_pid=pid)
                        self.logger.info(f"registered ckpt role-{pid} step {self._step} -> {ckpt_path}")
                    except Exception as exc: self.logger.info(f"add_checkpoint failed for role-{pid}: {exc}")

                # Only publish resumable state when every role has an adapter for
                # this iteration. A preemption between role updates therefore
                # falls back to the preceding complete, internally consistent step.
                if self._iteration_is_complete():
                    self._save_training_state()
                self._step += 1
            except Exception as exc:
                self.logger.exception(f"Exception in multirole learner loop: {exc}")

        self.logger.info("[MultiRoleLearner] training finished.")
        for pid in self.role_pids: self.buffers[pid].stop.remote()

    def _save_checkpoint(self, role_pid: int) -> pathlib.Path:
        # PeftModel.save_pretrained always writes each adapter into a subdirectory named after
        # the adapter (e.g. parent/role-0/{adapter_config.json,adapter_model.safetensors}).
        # vLLM's LoRARequest, however, expects adapter_config.json at the TOP LEVEL of the path
        # it's given. So save into the iteration dir (the parent), then return the inner
        # role-<pid> subdir as the vLLM-facing path.
        adapter_name = role_adapter_name(role_pid)
        iter_dir = self.ckpt_dir / f"iteration-{self._step}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        self.policy_model.set_adapter(adapter_name)
        try:
            # peft >= 0.7: only the requested adapter is written under iter_dir/<adapter_name>/
            self.policy_model.save_pretrained(iter_dir, selected_adapters=[adapter_name])
        except TypeError:
            # older peft: this writes ALL adapters into iter_dir/<each_adapter_name>/; the one
            # we care about is still iter_dir/<adapter_name>/, so the return path stays correct.
            self.policy_model.save_pretrained(iter_dir)
        return iter_dir / adapter_name

    def _iteration_is_complete(self) -> bool:
        iter_dir = self.ckpt_dir / f"iteration-{self._step}"
        return all(
            (iter_dir / role_adapter_name(pid) / "adapter_config.json").is_file()
            and (iter_dir / role_adapter_name(pid) / "adapter_model.safetensors").is_file()
            for pid in self.role_pids
        )

    @staticmethod
    def _cpu_gradient_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: param.grad.detach().cpu()
            for name, param in module.named_parameters()
            if param.grad is not None
        }

    @staticmethod
    def _cpu_trainable_parameter_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    def _save_training_state(self) -> pathlib.Path:
        """Atomically save the state needed to continue after a completed update.

        Adapter weights remain in PEFT's vLLM-compatible directories. AdamW
        moments, parameter gradients, counters, and torch RNG state live in one
        training_state.pt beside that iteration. Older training-state files are
        removed so optimizer checkpoints do not multiply disk usage.
        """
        iter_dir = self.ckpt_dir / f"iteration-{self._step}"
        state_path = iter_dir / "training_state.pt"
        tmp_path = iter_dir / f".{state_path.name}.{os.getpid()}.tmp"
        state = {
            "format_version": 1,
            "step": self._step,
            "samples_seen": dict(self._samples_seen),
            "policy_optimizers": {
                pid: optimizer.state_dict()
                for pid, optimizer in self.policy_optimizers.items()
            },
            "policy_gradients": self._cpu_gradient_state(self.policy_model),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
        critics = getattr(self, "critics", {})
        critic_optimizers = getattr(self, "critic_optimizers", {})
        if critics:
            # Critic base-model parameters are frozen, so save only trainable
            # LoRA/value-head tensors rather than duplicating the full backbone.
            state["critic_parameters"] = {
                pid: self._cpu_trainable_parameter_state(critic)
                for pid, critic in critics.items()
            }
            state["critic_gradients"] = {
                pid: self._cpu_gradient_state(critic)
                for pid, critic in critics.items()
            }
            state["critic_optimizers"] = {
                pid: optimizer.state_dict()
                for pid, optimizer in critic_optimizers.items()
            }

        torch.save(state, tmp_path)
        os.replace(tmp_path, state_path)

        # Keep exactly the newest complete optimizer/gradient state. The adapter
        # history stays untouched and can still be used for evaluation.
        for old_path in self.ckpt_dir.glob("iteration-*/training_state.pt"):
            if old_path != state_path:
                try:
                    old_path.unlink()
                except OSError as exc:
                    self.logger.warning(f"could not remove old training state {old_path}: {exc}")
        self.logger.info(f"saved resumable training state -> {state_path}")
        return state_path

    def _restore_training_state(self) -> None:
        if self._training_state_restored:
            return
        self._training_state_restored = True
        if not self._initial_training_state_path:
            return

        state_path = pathlib.Path(self._initial_training_state_path)
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        if state.get("format_version") != 1:
            raise RuntimeError(f"unsupported training-state format in {state_path}")
        expected_step = self._step - 1
        if int(state["step"]) != expected_step:
            raise RuntimeError(
                f"training state {state_path} is for step {state['step']}, "
                f"but resume expects completed step {expected_step}"
            )

        optimizer_states = state["policy_optimizers"]
        for pid, optimizer in self.policy_optimizers.items():
            saved = optimizer_states.get(pid, optimizer_states.get(str(pid)))
            if saved is None:
                raise RuntimeError(f"training state has no optimizer for role-{pid}")
            optimizer.load_state_dict(saved)

        critics = getattr(self, "critics", {})
        critic_optimizer_states = state.get("critic_optimizers", {})
        critic_parameter_states = state.get("critic_parameters", {})
        critic_gradient_states = state.get("critic_gradients", {})
        for pid, critic in critics.items():
            saved_parameters = critic_parameter_states.get(pid, critic_parameter_states.get(str(pid)))
            saved_optimizer = critic_optimizer_states.get(pid, critic_optimizer_states.get(str(pid)))
            if saved_parameters is None or saved_optimizer is None:
                raise RuntimeError(f"training state has no critic state for role-{pid}")
            named_critic_parameters = dict(critic.named_parameters())
            with torch.no_grad():
                for name, value in saved_parameters.items():
                    if name not in named_critic_parameters:
                        raise RuntimeError(f"critic role-{pid} has no saved parameter {name!r}")
                    named_critic_parameters[name].copy_(value)
            getattr(self, "critic_optimizers")[pid].load_state_dict(saved_optimizer)
            saved_gradients = critic_gradient_states.get(pid, critic_gradient_states.get(str(pid), {}))
            for name, gradient in saved_gradients.items():
                if name in named_critic_parameters:
                    named_critic_parameters[name].grad = gradient.to(
                        device=named_critic_parameters[name].device,
                        dtype=named_critic_parameters[name].dtype,
                    )

        named_parameters = dict(self.policy_model.named_parameters())
        for name, gradient in state.get("policy_gradients", {}).items():
            if name in named_parameters:
                named_parameters[name].grad = gradient.to(
                    device=named_parameters[name].device,
                    dtype=named_parameters[name].dtype,
                )
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng_state_all"]])
        self._samples_seen = {
            pid: int(state["samples_seen"].get(pid, state["samples_seen"].get(str(pid), 0)))
            for pid in self.role_pids
        }
        self.logger.info(f"restored optimizer, gradients, counters, and RNG from {state_path}")
