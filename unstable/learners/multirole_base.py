import ray, torch, time, pathlib
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
                 use_trainer_cache: bool=False, initial_lora_paths: Optional[Dict[int, str]]=None):
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
        if self.gradient_checkpointing:   self.policy_model.gradient_checkpointing_enable()
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

        self._step = 1; self._samples_seen: Dict[int, int] = {pid: 0 for pid in self.role_pids}

    def initialize_algorithm(self, *args, **kwargs): raise NotImplementedError
    def _update(self, role_pid: int, batch):         raise NotImplementedError

    def _ready_roles(self) -> List[int]:
        ready = []
        for pid in self.role_pids:
            if ray.get(self.buffers[pid].size.remote()) >= self.batch_size * 1.5:
                ready.append(pid)
        return ready

    def train(self, iterations: int):
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
