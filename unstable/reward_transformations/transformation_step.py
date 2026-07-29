from typing import List
from unstable._types import PlayerTrajectory

class StepRewardTransform:
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float: raise NotImplementedError

class ComposeStepRewardTransforms:
    def __init__(self, transforms: List[StepRewardTransform]): self.transforms = transforms
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        for transform in self.transforms: reward = transform(player_traj, step_index, reward)
        return reward

    def apply_with_components(self, player_traj: PlayerTrajectory, step_index: int, reward: float):
        """Apply transforms while exposing each additive contribution for diagnostics."""
        components = {}
        for transform in self.transforms:
            before = reward
            reward = transform(player_traj, step_index, reward)
            name = {
                "RewardForFormat": "format_reward",
                "PenaltyForInvalidMove": "invalid_move_reward",
                "EnvStepReward": "environment_reward",
            }.get(type(transform).__name__, type(transform).__name__)
            components[name] = components.get(name, 0.0) + float(reward - before)
        return reward, components

class RewardForFormat(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        feedback = player_traj.format_feedbacks[step_index]
        # When an environment reports phase-specific validation, require both
        # the outer \boxed{...} convention and a valid payload for that phase.
        # Legacy environments only provide correct_answer_format and retain the
        # previous behavior.
        valid = bool(feedback.get("correct_answer_format"))
        if "phase_format_valid" in feedback:
            valid = valid and bool(feedback["phase_format_valid"])
        reward += (self.reward if valid else self.penalty)
        return reward

class PenaltyForInvalidMove(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.penalty if player_traj.format_feedbacks[step_index].get("invalid_move") else self.reward)
        return reward

class EnvStepReward(StepRewardTransform):
    """Adds per-step reward supplied by the env (via state.step_info["step_rewards_by_pid"]).
    Missing/empty step_rewards on the trajectory ⇒ no-op — safe for envs that don't opt in."""
    def __init__(
        self,
        scale: float = 1.0,
        payoff_scale: float | None = None,
        prediction_scale: float | None = None,
    ):
        self.scale = scale
        self.component_scales = {
            "payoff": scale if payoff_scale is None else payoff_scale,
            "prediction": scale if prediction_scale is None else prediction_scale,
        }

    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        components = getattr(player_traj, "step_reward_components", None)
        if components and step_index < len(components) and components[step_index]:
            reward += sum(
                self.component_scales.get(name, self.scale) * float(value)
                for name, value in components[step_index].items()
            )
            return reward
        sr = getattr(player_traj, "step_rewards", None)
        if sr and step_index < len(sr):
            reward += self.scale * float(sr[step_index])
        return reward
