"""Checks for the matched-rollout no-prediction IPD control."""

import textarena as ta


ENV_ID = "IteratedPrisonersDilemma-Broadcast-v0-train"


def _finish_round(env, action0, action1):
    for action in (
        "[Message: test message 0]",
        "[Message: test message 1]",
        f"[Action: {action0}]",
        f"[Action: {action1}]",
    ):
        done, info = env.step(action)
    return done, info


def test_control_has_only_conversation_and_decision_phases():
    env = ta.make(ENV_ID)
    env.reset(num_players=2, seed=0)
    observed_phases = []
    turns = 0
    while True:
        phase = env.state.game_state["phase"]
        observed_phases.append(phase)
        pid, obs = env.get_observation()
        assert "PREDICTION" not in obs
        if phase == "conversation":
            action = f"[Message: role {pid} cooperates]"
        else:
            assert phase == "decision"
            action = "[Action: Cooperate]"
        done, info = env.step(action)
        turns += 1
        assert "prediction_rewards_by_pid" not in info
        if done:
            break

    assert set(observed_phases) == {"conversation", "decision"}
    assert turns == 40


def test_control_payoff_matrix_matches_prediction_task_payoffs():
    expected = {
        ("Cooperate", "Cooperate"): {0: 3.0, 1: 3.0},
        ("Defect", "Defect"): {0: 1.0, 1: 1.0},
        ("Cooperate", "Defect"): {0: 0.0, 1: 5.0},
        ("Defect", "Cooperate"): {0: 5.0, 1: 0.0},
    }
    for actions, payoffs in expected.items():
        control = ta.make(ENV_ID)
        control.reset(num_players=2, seed=0)
        _, control_info = _finish_round(control, *actions)

        predict = ta.make("IteratedPrisonersDilemma-Predict-v0-train")
        predict.reset(num_players=2, seed=0)
        for action in (
            "[Message: test message 0]",
            "[Message: test message 1]",
            "[Prediction: Cooperate]",
            "[Prediction: Cooperate]",
            f"[Action: {actions[0]}]",
            f"[Action: {actions[1]}]",
        ):
            _, predict_info = predict.step(action)

        assert control_info["step_rewards_by_pid"] == payoffs
        assert predict_info["step_rewards_by_pid"] == payoffs


def test_control_broadcast_is_simultaneous_and_context_is_retained():
    env = ta.make(ENV_ID)
    env.reset(num_players=2, seed=0)
    env.get_observation()
    env.step("[Message: private-until-both-submit]")
    pending = "\n".join(message for _, message, _ in env.state.observations[1])
    assert "private-until-both-submit" not in pending

    env.get_observation()
    env.step("[Message: second-message]")
    revealed = "\n".join(message for _, message, _ in env.state.observations[0])
    assert "private-until-both-submit" in revealed
    assert "second-message" in revealed

    env.step("[Action: Cooperate]")
    env.step("[Action: Cooperate]")
    env.step("[Message: round-two-message-0]")
    env.step("[Message: round-two-message-1]")
    env.step("[Action: Cooperate]")
    pid, obs = env.get_observation()
    assert pid == 1
    assert "private-until-both-submit" in obs
    assert "Round 1 results:" in obs
    assert "round-two-message-0" in obs
    assert "CURRENT PHASE: DECISION (round 2)" in obs


if __name__ == "__main__":
    test_control_has_only_conversation_and_decision_phases()
    test_control_payoff_matrix_matches_prediction_task_payoffs()
    test_control_broadcast_is_simultaneous_and_context_is_retained()
    print("OK")
