"""Smoke test for IteratedPrisonersDilemma-Predict-v0.

Exercises:
  - simultaneous broadcast comm (opponent should NOT see your message until both submit)
  - prediction phase with private prompts
  - decision payoff and prediction bonus emitted as separate reward payloads
"""
import textarena as ta


def _obs_texts(env, pid: int):
    """Peek at the pending observation queue for `pid` without draining it."""
    logs = env.state.observations.get(pid, [])
    return [msg for (_from, msg, _typ) in logs]


def test_ipd_predict():
    env = ta.make("IteratedPrisonersDilemma-Predict-v0")
    env.reset(num_players=2, seed=0)

    # Round 1, conversation turn 1 (communication_turns=1 in registration)
    pid, _ = env.get_observation()
    assert pid == 0
    env.step("some private reasoning {let's cooperate}")

    # After pid 0 speaks, pid 1 must NOT see pid 0's public message yet
    p1_before = "\n".join(_obs_texts(env, 1))
    assert "let's cooperate" not in p1_before, "leaked message before broadcast"

    pid, _ = env.get_observation()
    assert pid == 1
    env.step("{ok, cooperating}")

    # After both submit, both should see the aggregated broadcast
    p0_after = "\n".join(_obs_texts(env, 0))
    p1_after = "\n".join(_obs_texts(env, 1))
    assert "let's cooperate" in p0_after and "let's cooperate" in p1_after
    assert "ok, cooperating" in p0_after and "ok, cooperating" in p1_after

    # Phase should now be prediction
    assert env.state.game_state["phase"] == "prediction"

    # Prediction turn — pid 0 predicts pid 1 will cooperate
    pid, _ = env.get_observation()
    assert pid == 0
    env.step("<Cooperate>")

    # pid 1 should NOT see any PLAYER_ACTION from pid 0 during the prediction phase.
    # (The prediction PROMPT to pid 1 may mention "<Cooperate>" as an example — that's
    # a GAME_BOARD message, not a leak of pid 0's chosen prediction.)
    from textarena.core import ObservationType
    leaks = [(frm, msg) for (frm, msg, typ) in env.state.observations[1]
             if typ == ObservationType.PLAYER_ACTION and frm == 0]
    assert not leaks, f"pid 1 saw pid 0's PLAYER_ACTION: {leaks}"
    # And pid 0's own action IS echoed back to pid 0 (own PLAYER_ACTION):
    p0_actions = [msg for (frm, msg, typ) in env.state.observations[0]
                  if typ == ObservationType.PLAYER_ACTION and frm == 0]
    assert any("<Cooperate>" in m for m in p0_actions)

    pid, _ = env.get_observation()
    assert pid == 1
    env.step("<Defect>")   # pid 1 predicts pid 0 will defect (wrong)

    assert env.state.game_state["phase"] == "decision"

    # Decision turn — both cooperate
    pid, _ = env.get_observation()
    env.step("[Cooperate]")
    pid, _ = env.get_observation()
    done, step_info = env.step("[Cooperate]")

    # This is the step that triggered resolution — payoffs go to decision steps,
    # while prediction bonuses are separately routed to prediction steps by UB.
    assert "step_rewards_by_pid" in step_info, f"missing step_rewards_by_pid: {step_info}"
    assert "prediction_rewards_by_pid" in step_info, f"missing prediction rewards: {step_info}"
    srp = step_info["step_rewards_by_pid"]
    prp = step_info["prediction_rewards_by_pid"]
    assert set(srp.keys()) == {0, 1}
    # payoff for mutual cooperate = 3; pid 0 predicted correctly (+1), pid 1 wrong (+0)
    assert srp[0] == 3.0, f"expected 3.0, got {srp[0]}"
    assert srp[1] == 3.0, f"expected 3.0, got {srp[1]}"
    assert prp == {0: 1.0, 1: 0.0}, prp


def test_ipd_predict_retry_still_resolves_shaped_rewards():
    """A repairable format miss must not suppress the round's reward signal."""
    env = ta.make("IteratedPrisonersDilemma-Predict-v0-train")
    env.reset(num_players=2, seed=0)
    env.state.error_allowance = 2

    # Finish simultaneous communication and prediction.
    for action in ("{cooperate}", "{cooperate}", "<Cooperate>", "<Cooperate>"):
        done, _ = env.step(action)
        assert not done

    # Player 0 repairs one malformed decision, then both decisions resolve.
    done, info = env.step("not a valid decision")
    assert not done
    assert info["phase_format_valid_by_pid"] == {0: False}
    done, _ = env.step("[Cooperate]")
    assert not done
    done, info = env.step("[Cooperate]")
    assert not done
    assert info["step_rewards_by_pid"] == {0: 3.0, 1: 3.0}
    assert info["prediction_rewards_by_pid"] == {0: 1.0, 1: 1.0}


if __name__ == "__main__":
    test_ipd_predict()
    print("OK")
