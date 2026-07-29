from collections import Counter
import unittest

from unstable._types import PlayerTrajectory, Step
from unstable.buffers import _sample_balanced_steps, _terminal_reward
from unstable.learners.multirole_reinforce import _validated_phase_loss_weights
from unstable.reward_transformations.transformation_final import RoleAdvantageByEnvFormatter
from unstable.reward_transformations import (
    ComposeStepRewardTransforms,
    EnvStepReward,
    NormalizeRewardsByEnvPhase,
    RewardForFormat,
)


PHASES = ("conversation", "prediction", "decision")


def _step(phase, reward, index):
    return Step(
        pid=0, obs=f"obs-{index}", act=f"act-{index}", reward=reward,
        env_id="ipd", step_info={}, phase=phase,
        reward_components={"reward_before_normalization": reward},
    )


class PhaseBalancedReinforceTests(unittest.TestCase):
    def test_terminal_reward_can_be_disabled_without_calling_transform(self):
        class MustNotRun:
            def __call__(self, **kwargs):
                raise AssertionError("terminal transform was called")

        self.assertEqual(_terminal_reward(False, 1.0, 0, "ipd", MustNotRun()), 0.0)

    def test_approved_phase_reward_contract(self):
        traj = PlayerTrajectory(
            format_feedbacks=[
                {"correct_answer_format": True, "phase_format_valid": True},
                {"correct_answer_format": True, "phase_format_valid": True},
                {"correct_answer_format": True, "phase_format_valid": True},
            ],
            step_rewards=[0.0, 1.0, 3.0],
        )
        transforms = ComposeStepRewardTransforms([RewardForFormat(1.5), EnvStepReward(1.0)])
        self.assertEqual([transforms(traj, index, 0.0) for index in range(3)], [1.5, 2.5, 4.5])

    def test_payoff_and_prediction_scales_are_independent(self):
        traj = PlayerTrajectory(
            step_rewards=[3.0, 1.0],
            step_reward_components=[
                {"payoff": 3.0},
                {"prediction": 1.0},
            ],
        )
        transform = EnvStepReward(
            scale=9.0,
            payoff_scale=0.5,
            prediction_scale=2.0,
        )
        self.assertEqual(transform(traj, 0, 0.0), 1.5)
        self.assertEqual(transform(traj, 1, 0.0), 2.0)

    def test_legacy_env_step_scale_remains_supported(self):
        traj = PlayerTrajectory(step_rewards=[4.0])
        self.assertEqual(EnvStepReward(0.25)(traj, 0, 0.0), 1.0)

    def test_terminal_reward_scale_is_independent(self):
        transform = RoleAdvantageByEnvFormatter(tau=0.0, scale=0.5)
        self.assertEqual(transform(reward=4.0, pid=0, env_id="ipd"), 2.0)

    def test_balanced_sampling_uses_equal_unique_phase_samples(self):
        steps = [_step(phase, float(index), index) for phase in PHASES for index in range(10)]
        batch = _sample_balanced_steps(steps, batch_size=12, phases=PHASES)
        self.assertEqual(Counter(step.phase for step in batch), Counter({phase: 4 for phase in PHASES}))
        self.assertEqual(len({id(step) for step in batch}), len(batch))

    def test_balanced_sampling_refuses_to_duplicate_scarce_phase(self):
        steps = [_step("conversation", 0.0, i) for i in range(4)]
        with self.assertRaisesRegex(ValueError, "prediction"):
            _sample_balanced_steps(steps, batch_size=6, phases=PHASES)

    def test_matched_control_samples_128_from_each_shared_phase(self):
        phases = ("conversation", "decision")
        steps = [
            _step(phase, float(index), index)
            for phase in phases
            for index in range(256)
        ]
        batch = _sample_balanced_steps(steps, batch_size=256, phases=phases)
        self.assertEqual(
            Counter(step.phase for step in batch),
            Counter({"conversation": 128, "decision": 128}),
        )
        self.assertEqual(len({id(step) for step in batch}), 256)

    def test_phase_local_normalization_centers_each_phase_without_rescaling(self):
        steps = []
        for phase, rewards in {
            "conversation": [0.0, 2.0],
            "prediction": [10.0, 14.0],
            "decision": [-3.0, 3.0],
        }.items():
            steps.extend(_step(phase, reward, len(steps)) for reward in rewards)

        NormalizeRewardsByEnvPhase()(steps)
        grouped = {phase: [step for step in steps if step.phase == phase] for phase in PHASES}
        for phase_steps in grouped.values():
            self.assertAlmostEqual(sum(step.reward for step in phase_steps), 0.0)
        self.assertEqual([step.reward for step in grouped["prediction"]], [-2.0, 2.0])

    def test_phase_loss_weights_normalize_by_default(self):
        weights = _validated_phase_loss_weights(
            {"conversation": 1.0 / 3.0, "decision": 1.0 / 3.0}
        )
        self.assertEqual(weights, {"conversation": 0.5, "decision": 0.5})

    def test_phase_loss_weights_can_preserve_absolute_coefficients(self):
        weights = _validated_phase_loss_weights(
            {"conversation": 1.0 / 3.0, "decision": 1.0 / 3.0},
            normalize=False,
        )
        self.assertAlmostEqual(weights["conversation"], 1.0 / 3.0)
        self.assertAlmostEqual(weights["decision"], 1.0 / 3.0)
        self.assertAlmostEqual(sum(weights.values()), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
