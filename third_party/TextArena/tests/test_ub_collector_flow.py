"""End-to-end test of the UB collector -> trajectory step_rewards flow.

Simulates what `unstable.collector.run_game` does with a scripted (non-vLLM) actor,
so we can verify:
  - traj.step_rewards has one float per step
  - env-provided step_rewards_by_pid lands on each pid's most-recent step
  - EnvStepReward transform correctly propagates it into Step.reward
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../..")

import textarena as ta
from unstable._types import PlayerTrajectory
from unstable.reward_transformations.transformation_step import EnvStepReward, RewardForFormat
from unstable.utils.templates import apply_template, extract_action_and_format_feedback


def _apply_env_rewards(trajs, step_info):
    """Mirror collector reward routing, including deferred prediction bonuses."""
    for tgt, r in ((step_info or {}).get("step_rewards_by_pid") or {}).items():
        if trajs[tgt].step_rewards:
            trajs[tgt].step_rewards[-1] += float(r)
    for tgt, r in ((step_info or {}).get("prediction_rewards_by_pid") or {}).items():
        for i in range(len(trajs[tgt].step_phases) - 1, -1, -1):
            if trajs[tgt].step_phases[i] == "prediction":
                trajs[tgt].step_rewards[i] += float(r)
                break


def run_scripted_game(env_id: str, actions_by_pid: dict, num_players: int):
    """Mimic collector.run_game but with hard-coded actions per phase index."""
    env = ta.make(env_id); env.reset(num_players=num_players, seed=0)
    trajs = {pid: PlayerTrajectory(pid=pid) for pid in range(num_players)}
    turn_counters = {pid: 0 for pid in range(num_players)}
    while True:
        pid, obs = env.get_observation()
        act = actions_by_pid[pid][turn_counters[pid]]
        turn_counters[pid] += 1
        phase = env.state.game_state.get("phase")
        done, step_info = env.step(act)
        # per-turn tracking
        trajs[pid].obs.append(obs); trajs[pid].actions.append(act)
        trajs[pid].extracted_actions.append(act)
        trajs[pid].format_feedbacks.append({"correct_answer_format": True, "invalid_move": False})
        trajs[pid].step_infos.append(step_info)
        trajs[pid].step_rewards.append(0.0)
        trajs[pid].step_phases.append(phase)
        _apply_env_rewards(trajs, step_info)
        if done: break
    return trajs


def test_ipd_predict_collector_flow():
    # scripted 1-round IPD: both cooperate; pid 0 predicts C (correct), pid 1 predicts D (wrong)
    actions = {
        0: ["[Message: cooperate]", "[Prediction: Cooperate]", "[Action: Cooperate]"],
        1: ["[Message: cooperate]", "[Prediction: Defect]",    "[Action: Cooperate]"],
    }
    # IPD-Predict-v0 default num_rounds=10 comm_turns=1 — but we only script 1 round.
    # To keep the game short, monkey-patch num_rounds down after reset via game_state.
    env = ta.make("IteratedPrisonersDilemma-Predict-v0")
    env.reset(num_players=2, seed=0)
    env.state.game_state["num_rounds"] = 1  # end after this round

    from unstable._types import PlayerTrajectory
    trajs = {pid: PlayerTrajectory(pid=pid) for pid in range(2)}
    turn = {0: 0, 1: 0}
    while True:
        pid, obs = env.get_observation()
        act = actions[pid][turn[pid]]; turn[pid] += 1
        phase = env.state.game_state.get("phase")
        done, step_info = env.step(act)
        trajs[pid].obs.append(obs); trajs[pid].actions.append(act)
        trajs[pid].extracted_actions.append(act)
        trajs[pid].format_feedbacks.append({"correct_answer_format": True, "invalid_move": False})
        trajs[pid].step_infos.append(step_info)
        trajs[pid].step_rewards.append(0.0)
        trajs[pid].step_phases.append(phase)
        _apply_env_rewards(trajs, step_info)
        if done: break

    # Each pid took 3 steps (comm, pred, decision). Prediction reward belongs
    # to prediction; payoff belongs to decision.
    assert len(trajs[0].step_rewards) == 3
    assert trajs[0].step_rewards[0] == 0.0
    assert trajs[0].step_rewards[1] == 1.0, trajs[0].step_rewards
    assert trajs[0].step_rewards[2] == 3.0, trajs[0].step_rewards
    assert trajs[1].step_rewards[2] == 3.0, trajs[1].step_rewards  # 3 (mut coop) + 0 (wrong pred)

    # EnvStepReward transform applies scale
    xform = EnvStepReward(scale=1.0)
    for pid in (0, 1):
        for i, sr in enumerate(trajs[pid].step_rewards):
            got = xform(trajs[pid], i, 0.0)
            assert got == sr, (pid, i, sr, got)


def test_boxed_payload_is_preserved():
    assert extract_action_and_format_feedback(r"\boxed{{hello}}") == ("{hello}", {"correct_answer_format": True})
    assert extract_action_and_format_feedback(r"\boxed{<Cooperate>}") == ("<Cooperate>", {"correct_answer_format": True})
    assert extract_action_and_format_feedback(r"\boxed{[Cooperate]}") == ("[Cooperate]", {"correct_answer_format": True})
    assert extract_action_and_format_feedback(
        r"\boxed{[Prediction: Cooperate]}"
    ) == ("[Prediction: Cooperate]", {"correct_answer_format": True})
    traj = PlayerTrajectory(format_feedbacks=[
        {"correct_answer_format": True, "phase_format_valid": True},
        {"correct_answer_format": True, "phase_format_valid": False},
    ])
    xform = RewardForFormat(reward=1.5, penalty=0.0)
    assert xform(traj, 0, 0.0) == 1.5
    assert xform(traj, 1, 0.0) == 0.0


def test_multiphase_template_treats_required_output_as_boxed_payload():
    prompt = apply_template(
        "qwen3-multiphase",
        "CURRENT PHASE: DECISION (round 1).\n"
        "REQUIRED OUTPUT: Reply with exactly one command: "
        "[Action: Cooperate] or [Action: Defect]",
    )
    assert "REQUIRED OUTPUT describes only that payload" in prompt
    assert r"\boxed{[Action: Cooperate]}" in prompt
    assert "Never emit the command bare" in prompt


if __name__ == "__main__":
    test_ipd_predict_collector_flow()
    test_boxed_payload_is_preserved()
    test_multiphase_template_treats_required_output_as_boxed_payload()
    print("OK")
