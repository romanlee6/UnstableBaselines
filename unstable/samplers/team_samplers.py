import random
import ray
from typing import Dict, Any, List, Optional

from unstable._types import AgentSpec, GameInformation


class FixedRoleTeamSampler:
    """Assembles a team of role-LoRAs across env seats.

    Set `shuffle_roles=True` (default) to randomize which role's LoRA plays which
    env pid per training episode. This is the mitigation for env-level positional
    asymmetry (e.g. IPD's leaky conversation phase where pid=1 reads pid=0's
    message before responding): with shuffle, each role's data-collection is
    averaged over both seats, so LoRAs do not bake in a positional advantage.

    Buffer routing is by role_pid (not env pid), so shuffling seats does not
    misroute trajectories.

    Train: every seat runs a role-LoRA checkpoint; all trajectories are collected.
    Eval:  configurable per pid - either the trained LoRA or an OpenRouter substitute
           via `eval_substitutions = {pid: openrouter_name}`. Eval never shuffles
           (fixed_opponent semantics require deterministic seat assignment).
    """

    def __init__(
        self,
        model_registry,
        role_pids: List[int],
        eval_substitutions: Optional[Dict[int, str]] = None,
        eval_provider: str = "openrouter",
        shuffle_roles: bool = True,
    ):
        self.model_registry = model_registry
        self.role_pids = list(role_pids)
        self.eval_substitutions = dict(eval_substitutions or {})
        self.eval_provider = eval_provider
        self.shuffle_roles = shuffle_roles

        # ensure each OpenRouter substitute exists as a fixed entry so rating updates work
        for _, openrouter_name in self.eval_substitutions.items():
            self.model_registry.add_fixed.remote(name=openrouter_name)

    def _ckpt_for_role(self, pid: int):
        uid = ray.get(self.model_registry.get_current_ckpt_by_role.remote(role_pid=pid))
        if uid is None: raise RuntimeError(f"no current checkpoint registered for role pid={pid}")
        path = ray.get(self.model_registry.get_name_or_lora_path.remote(uid=uid))
        return uid, path

    def sample_train_team(self, env_spec):
        assert env_spec.num_players == len(self.role_pids), (
            f"FixedRoleTeamSampler expects num_players ({env_spec.num_players}) "
            f"== len(role_pids) ({len(self.role_pids)}) so every seat gets one role."
        )
        seat_to_role = list(self.role_pids)
        if self.shuffle_roles:
            random.shuffle(seat_to_role)

        agent_specs, models = [], []
        for seat_pid, role_pid in enumerate(seat_to_role):
            uid, lora_path = self._ckpt_for_role(role_pid)
            agent_specs.append(AgentSpec(
                pid=seat_pid, kind="checkpoint", collect_data=True, lora_path=lora_path,
                prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn,
                role_pid=role_pid, model_uid=uid,
            ))
            models.append({"uid": uid, "pid": seat_pid, "type": "model", "role_pid": role_pid, "source": "checkpoint"})
        return agent_specs, models

    def sample_eval_team(self, env_spec):
        agent_specs, models = [], []
        for pid in range(env_spec.num_players):
            if pid in self.eval_substitutions:
                openrouter_name = self.eval_substitutions[pid]
                agent_specs.append(AgentSpec(
                    pid=pid, kind=self.eval_provider, lora_path=None, openrouter_name=openrouter_name,
                    external_provider=self.eval_provider,
                    model_uid=f"fixed-{openrouter_name}",
                ))
                models.append({"uid": f"fixed-{openrouter_name}", "pid": pid, "type": "opponent", "role_pid": pid, "source": self.eval_provider})
            else:
                assert pid in self.role_pids, f"pid {pid} not assigned to any role and not substituted"
                uid, lora_path = self._ckpt_for_role(pid)
                agent_specs.append(AgentSpec(
                    pid=pid, kind="checkpoint", collect_data=False, lora_path=lora_path,
                    prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn,
                    model_uid=uid,
                ))
                models.append({"uid": uid, "pid": pid, "type": "model", "role_pid": pid, "source": "checkpoint"})
        return agent_specs, models

    def update(self, game_info: GameInformation, job_info: Dict[str, Any]):
        uids, scores = [], []
        for m in job_info["models"]:
            if m["pid"] in game_info.final_rewards:
                uids.append(m["uid"]); scores.append(game_info.final_rewards[m["pid"]])
        if uids:
            self.model_registry.update_ratings.remote(uids=uids, scores=scores, env_id=job_info["env_id"])
