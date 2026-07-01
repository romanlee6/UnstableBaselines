import numpy as np
from typing import List, Optional
from collections import defaultdict
from unstable._types import Step

class SamplingRewardTransform:
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]: raise NotImplementedError

class ComposeSamplingRewardTransforms:
    def __init__(self, transforms: List[SamplingRewardTransform]):  self.transforms = transforms
    def __call__(self, steps: List[Step]) -> List[Step]:
        for transform in self.transforms: steps = transform(steps)
        return steps

class NormalizeRewards(SamplingRewardTransform):
    def __init__(self, z_score: bool=False): self.z_score = z_score
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        rewards = [step.reward for step in steps]
        mean, std = np.mean(rewards), np.std(rewards)+1e-8
        for step in steps: step.reward = (step.reward-mean)/(std if self.z_score else 1)
        return steps

class NormalizeRewardsByEnv(SamplingRewardTransform):
    def __init__(self, z_score: bool = False): self.z_score = z_score
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        env_buckets = defaultdict(list)
        for step in steps: env_buckets[step.env_id].append(step) # bucket by env
        for env_steps in env_buckets.values():
            r = np.asarray([s.reward for s in env_steps], dtype=np.float32)
            normed = ((r-r.mean())/(r.std()+1e-8)) if self.z_score else r-r.mean()
            for s, nr in zip(env_steps, normed): s.reward = float(nr) # write back
        return steps

class NormalizeAdvantagesByPidEnv(SamplingRewardTransform):
    # Per-(env_id, pid) z-score (or mean-subtract) at batch-sampling time. In multirole
    # each role's buffer holds a single pid so the grouping collapses to per-env; in
    # single-role training with a mixed-pid batch the (env_id, pid) key is meaningful.
    def __init__(self, z_score: bool = True): self.z_score = z_score
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        buckets = defaultdict(list)
        for s in steps: buckets[(s.env_id, s.pid)].append(s)
        for grp in buckets.values():
            r = np.asarray([s.reward for s in grp], dtype=np.float32)
            normed = ((r - r.mean()) / (r.std() + 1e-8)) if self.z_score else r - r.mean()
            for s, nr in zip(grp, normed): s.reward = float(nr)
        return steps

