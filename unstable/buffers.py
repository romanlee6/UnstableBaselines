
import os, ray, tree, random
from collections import Counter
from threading import Lock
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from unstable.utils.logging import setup_logger
from unstable._types import PlayerTrajectory, Step
# from unstable.core import BaseTracker
from unstable.trackers import BaseTracker
from unstable.utils import write_training_data_to_file
from unstable.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms


def _balanced_phase_quota(batch_size: int, phases: Tuple[str, ...]) -> int:
    if not phases or batch_size % len(phases) != 0:
        raise ValueError("batch_size must be divisible by the number of balanced phases")
    return batch_size // len(phases)


def _sample_balanced_steps(steps: List[Step], batch_size: int, phases: Tuple[str, ...]) -> List[Step]:
    quota = _balanced_phase_quota(batch_size, phases)
    batch = []
    for phase in phases:
        candidates = [step for step in steps if step.phase == phase]
        if len(candidates) < quota:
            raise ValueError(f"phase {phase!r} has {len(candidates)} samples; need {quota}")
        batch.extend(random.sample(candidates, quota))
    random.shuffle(batch)
    return batch


def _terminal_reward(include_final_reward, final_reward, pid, env_id, transformation):
    if not include_final_reward:
        return 0.0
    return transformation(reward=final_reward, pid=pid, env_id=env_id) if transformation else final_reward


class BaseBuffer:
    def __init__(self, max_buffer_size: int, tracker: BaseTracker, final_reward_transformation: Optional[ComposeFinalRewardTransforms], step_reward_transformation: Optional[ComposeStepRewardTransforms], sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], buffer_strategy: str = "random"): ...
    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str): ...
    def get_batch(self, batch_size: int): ...



@ray.remote
class StepBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker,
        final_reward_transformation: Optional[ComposeFinalRewardTransforms],
        step_reward_transformation: Optional[ComposeStepRewardTransforms],
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms],
        buffer_strategy: str = "random",
        role_pid: Optional[int] = None,
        include_final_reward: bool = True,
        balanced_phases: Optional[Tuple[str, ...]] = None,
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.steps: List[Step] = []
        self.training_steps = 0
        self.tracker = tracker
        self.role_pid = role_pid
        self.include_final_reward = include_final_reward
        self.balanced_phases = tuple(balanced_phases or ())
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote())) # setup logging
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        terminal_reward = _terminal_reward(
            self.include_final_reward, player_traj.final_reward, player_traj.pid,
            env_id, self.final_reward_transformation,
        )
        for idx in range(len(player_traj.obs)):
            phase = player_traj.step_phases[idx] if idx < len(player_traj.step_phases) else None
            components = {"terminal_reward": float(terminal_reward)}
            if self.step_reward_transformation and hasattr(self.step_reward_transformation, "apply_with_components"):
                step_reward, additions = self.step_reward_transformation.apply_with_components(
                    player_traj=player_traj, step_index=idx, reward=terminal_reward,
                )
                components.update(additions)
            else:
                step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=terminal_reward) if self.step_reward_transformation else terminal_reward
            components["reward_before_normalization"] = float(step_reward)
            feedback = player_traj.format_feedbacks[idx]
            components.update({
                "outer_format_valid": float(bool(feedback.get("correct_answer_format"))),
                "payload_format_valid": float(bool(feedback.get("phase_format_valid", True))),
                "joint_format_valid": float(bool(feedback.get("correct_answer_format")) and bool(feedback.get("phase_format_valid", True))),
            })
            with self.mutex:
                self.steps.append(Step(
                    pid=player_traj.pid, obs=player_traj.obs[idx], act=player_traj.actions[idx],
                    reward=step_reward, env_id=env_id,
                    step_info={"raw_reward": player_traj.final_reward, "env_reward": terminal_reward, "step_reward": step_reward, "phase": phase},
                    game_idx=player_traj.game_idx, role_pid=player_traj.role_pid,
                    own_model_uid=player_traj.own_model_uid, opponent_model_uids=player_traj.opponent_model_uids,
                    phase=phase, reward_components=components,
                ))
        self.logger.info(f"Buffer size: {len(self.steps)}, added {len(player_traj.obs)} steps")
        # downsample if necessary
        with self.mutex:
            if self.balanced_phases:
                capacity = self.max_buffer_size // len(self.balanced_phases)
                for phase in self.balanced_phases:
                    phase_steps = [step for step in self.steps if step.phase == phase]
                    for step in random.sample(phase_steps, max(0, len(phase_steps) - capacity)):
                        self.steps.remove(step)
            else:
                for step in random.sample(self.steps, max(0, len(self.steps) - self.max_buffer_size)):
                    self.steps.remove(step)
            counts = Counter(step.phase for step in self.steps)
        self.logger.info(f"Buffer size after downsampling: {len(self.steps)}; phase_counts={dict(counts)}")
        self.tracker.log_buffer.remote({"role_pid": self.role_pid, "total": len(self.steps), "phase_counts": dict(counts)})

    def get_batch(self, batch_size: int) -> List[Step]:
        with self.mutex: 
            if self.balanced_phases:
                batch = _sample_balanced_steps(self.steps, batch_size, self.balanced_phases)
            else:
                batch = random.sample(self.steps, batch_size)
            for b in batch: self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        self.logger.info(f"Sampling {len(batch)} samples from buffer.")
        suffix = f"_role_{self.role_pid}" if self.role_pid is not None else ""
        try: write_training_data_to_file(batch=batch, filename=os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}{suffix}.csv"))
        except Exception as exc: self.logger.error(f"Exception when trying to write training data to file: {exc}")
        self.training_steps += 1
        return batch

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(self.steps)
    def ready_for_batch(self, batch_size: int) -> bool:
        if not self.balanced_phases:
            return len(self.steps) >= batch_size * 1.5
        try: quota = _balanced_phase_quota(batch_size, self.balanced_phases)
        except ValueError: return False
        counts = Counter(step.phase for step in self.steps)
        return all(counts[phase] >= quota for phase in self.balanced_phases)
    def phase_counts(self) -> Dict[Optional[str], int]:
        return dict(Counter(step.phase for step in self.steps))
    def continue_collection(self):  return self.collect
    def clear(self):                
        with self.mutex: 
            self.steps.clear()


@ray.remote
class EpisodeBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker, 
        final_reward_transformation: Optional[ComposeFinalRewardTransforms], 
        step_reward_transformation: Optional[ComposeStepRewardTransforms], 
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], 
        buffer_strategy: str = "random"
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote()))  # setup logging
        self.episodes: List[List[Step]] = []
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        episode = []
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        for idx in range(len(player_traj.obs)):
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            episode.append(Step(
                pid=player_traj.pid, obs=player_traj.obs[idx], act=player_traj.actions[idx],
                reward=step_reward, env_id=env_id,
                step_info={"raw_reward": player_traj.final_reward, "env_reward": reward, "step_reward": step_reward},
                game_idx=player_traj.game_idx, role_pid=player_traj.role_pid,
                own_model_uid=player_traj.own_model_uid, opponent_model_uids=player_traj.opponent_model_uids,
            ))
        with self.mutex:
            self.episodes.append(episode)
            excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
            self.logger.info(f"BUFFER NUM of STEP {len(tree.flatten(self.episodes))}")
            while excess_num_samples > 0:
                randm_sampled = random.sample(self.episodes, 1)
                for b in randm_sampled: self.episodes.remove(b)
                excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
        
    def get_batch(self, batch_size: int) -> List[List[Step]]:
        with self.mutex:
            assert len(tree.flatten(self.episodes)) >= batch_size
            step_count = 0
            sampled_episodes = []
            random.shuffle(self.episodes)
            for ep in self.episodes:
                sampled_episodes.append(ep)
                step_count += len(ep)
                if step_count >= batch_size: break
            for ep in sampled_episodes: self.episodes.remove(ep)
        self.logger.info(f"Sampling {len(sampled_episodes)} episodes from buffer.")
        self.training_steps += 1
        return sampled_episodes

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(tree.flatten(self.episodes))
    def ready_for_batch(self, batch_size: int) -> bool: return self.size() >= batch_size * 1.5
    def continue_collection(self):  return self.collect
    def clear(self):
        with self.mutex: 
            self.episodes.clear()
