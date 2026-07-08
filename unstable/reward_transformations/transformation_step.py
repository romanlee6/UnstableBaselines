from typing import List
from unstable._types import PlayerTrajectory

class StepRewardTransform:
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float: raise NotImplementedError

class ComposeStepRewardTransforms:
    def __init__(self, transforms: List[StepRewardTransform]): self.transforms = transforms
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        for transform in self.transforms: reward = transform(player_traj, step_index, reward)
        return reward

class RewardForFormat(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.reward if player_traj.format_feedbacks[step_index].get("correct_answer_format") else self.penalty)
        return reward

class PenaltyForInvalidMove(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.penalty if player_traj.format_feedbacks[step_index].get("invalid_move") else self.reward)
        return reward

class EnvStepReward(StepRewardTransform):
    """Adds per-step reward supplied by the env (via state.step_info["step_rewards_by_pid"]).
    Missing/empty step_rewards on the trajectory ⇒ no-op — safe for envs that don't opt in."""
    def __init__(self, scale: float = 1.0): self.scale = scale
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        sr = getattr(player_traj, "step_rewards", None)
        if sr and step_index < len(sr):
            reward += self.scale * float(sr[step_index])
        return reward