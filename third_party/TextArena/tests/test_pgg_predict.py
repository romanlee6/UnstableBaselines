"""Smoke test for PublicGoodsGame-Predict-v0.

Exercises:
  - prediction phase with private prompts (aggregate pool prediction)
  - per-round payoff + prediction bonus written to state.step_info["step_rewards_by_pid"]
"""
import textarena as ta
from textarena.core import ObservationType


def test_pgg_predict():
    env = ta.make("PublicGoodsGame-Predict-v0")
    env.reset(num_players=3, seed=0)  # matches registered default

    N = 3
    endowment = env.state.game_state["endowment"]
    total_conv = env.state.game_state["total_conversation_rounds"]

    # Drive full conversation phase: total_conv * N steps of "{msg}"
    for _ in range(total_conv):
        for _ in range(N):
            pid, _ = env.get_observation()
            env.step(f"{{p{pid} says hi}}")

    assert env.state.game_state["phase"] == "prediction", env.state.game_state["phase"]

    # Prediction phase: each player privately predicts. Player 0 predicts 30, others 15.
    predictions = {0: 30, 1: 15, 2: 15}
    for _ in range(N):
        pid, _ = env.get_observation()
        env.step(f"<{predictions[pid]}>")

    # Verify privacy: no other player's PLAYER_ACTION from prediction should have leaked
    for pid in range(N):
        leaks = [(frm, msg) for (frm, msg, typ) in env.state.observations[pid]
                 if typ == ObservationType.PLAYER_ACTION and frm != pid]
        assert not leaks, f"pid {pid} saw another player's PLAYER_ACTION: {leaks}"

    assert env.state.game_state["phase"] == "decision"

    # Decision phase: contributions of 10 each -> total_contribution = 30
    contribs = {0: 10, 1: 10, 2: 10}
    step_info_final = None
    for _ in range(N):
        pid, _ = env.get_observation()
        done, step_info = env.step(f"[{contribs[pid]}]")
        step_info_final = step_info

    # The final decision step is the one that triggers resolution
    assert "step_rewards_by_pid" in step_info_final, f"missing step_rewards_by_pid: {step_info_final}"
    srp = step_info_final["step_rewards_by_pid"]
    assert set(srp.keys()) == {0, 1, 2}

    # Compute expected payoffs:
    # total = 30, public_good = 30 * 1.5 = 45, share = 45 / 3 = 15
    # each keeps endowment - contrib = 10, so payoff = 10 + 15 = 25
    # Prediction: total = 30, tolerance = endowment = 20, reward = 1.0
    #   pid 0 predicted 30 → error 0 → +1.0
    #   pid 1 predicted 15 → error 15 → +(1 - 15/20) = +0.25
    #   pid 2 predicted 15 → error 15 → +0.25
    expected = {0: 25.0 + 1.0, 1: 25.0 + 0.25, 2: 25.0 + 0.25}
    for pid, v in expected.items():
        assert abs(srp[pid] - v) < 1e-6, f"pid {pid} expected {v}, got {srp[pid]}"


if __name__ == "__main__":
    test_pgg_predict()
    print("OK")
