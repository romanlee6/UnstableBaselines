import copy
from typing import Dict, List, Tuple, Union, Optional

class FinalRewardTransform:
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float: raise NotImplementedError

class ComposeFinalRewardTransforms:
    def __init__(self, transforms: List[FinalRewardTransform]): self.transforms = transforms
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        for transform in self.transforms: reward = transform(reward, pid, env_id)
        return reward

class RoleAdvantageFormatter(FinalRewardTransform):
    def __init__(self, role_adv: float=0.0, tau: float=0.001): self.role_adv, self.tau = role_adv, tau
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        self.role_adv[pid] = (1-self.tau) * self.role_adv[pid] + self.tau * reward
        reward -= self.role_adv[pid]
        return reward

class RoleAdvantageByEnvFormatter(FinalRewardTransform):
    def __init__(self, default_role_adv: float=0.0, tau: float=0.001, scale: float=1.0):
        self.default_role_adv, self.tau, self.scale, self.role_advantage_dict = default_role_adv, tau, scale, {}
    def __call__(self, reward: float, pid: int, env_id: Optional[str] = None) -> float:
        if env_id not in self.role_advantage_dict: self.role_advantage_dict[env_id] = {}
        self.role_advantage_dict[env_id][pid] = (1-self.tau) *  self.role_advantage_dict[env_id].get(pid, self.default_role_adv) + self.tau * reward
        reward -= self.role_advantage_dict[env_id][pid]
        return self.scale * reward
