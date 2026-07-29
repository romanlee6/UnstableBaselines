"""Smoke test for PublicGoodsGame-Predict-v0.

Exercises:
  - prediction phase with private prompts (aggregate pool prediction)
  - decision payoff and prediction bonus emitted as separate reward payloads
"""
import textarena as ta
from textarena.core import ObservationType


def test_pgg_predict():
    env = ta.make("PublicGoodsGame-Predict-v0")
    env.reset(num_players=3, seed=0)  # matches registered default

    N = 3
    endowment = env.state.game_state["endowment"]
    total_conv = env.state.game_state["total_conversation_rounds"]

    # Drive full buffered conversation phase.
    for chat_round in range(total_conv):
        for _ in range(N):
            pid, _ = env.get_observation()
            env.step(f"[Message: p{pid} says hi]")
            if chat_round == 0 and pid == 0:
                assert "p0 says hi" not in "\n".join(
                    message for _, message, _ in env.state.observations[1]
                )

    assert env.state.game_state["phase"] == "prediction", env.state.game_state["phase"]

    # Each player predicts every opponent's individual contribution.
    for _ in range(N):
        pid, _ = env.get_observation()
        opponents = [other for other in range(N) if other != pid]
        predicted = {
            other: (0 if pid == 0 and other == 1 else 10)
            for other in opponents
        }
        env.step(
            "[Prediction: "
            + ", ".join(f"P{other}={predicted[other]}" for other in opponents)
            + "]"
        )

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
        done, step_info = env.step(f"[Action: {contribs[pid]}]")
        step_info_final = step_info

    # The final decision step is the one that triggers resolution
    assert "step_rewards_by_pid" in step_info_final, f"missing step_rewards_by_pid: {step_info_final}"
    assert "prediction_rewards_by_pid" in step_info_final, f"missing prediction rewards: {step_info_final}"
    srp = step_info_final["step_rewards_by_pid"]
    prp = step_info_final["prediction_rewards_by_pid"]
    assert set(srp.keys()) == {0, 1, 2}

    # Compute expected payoffs:
    # total = 30, public_good = 30 * 1.5 = 45, share = 45 / 3 = 15
    # each keeps endowment - contrib = 10, so payoff = 10 + 15 = 25
    # P0 has one exact prediction and one error of 10. With tolerance 20 its
    # averaged score is (1 + 0.5) / 2 = 0.75; all others are exact.
    expected = {0: 25.0, 1: 25.0, 2: 25.0}
    for pid, v in expected.items():
        assert abs(srp[pid] - v) < 1e-6, f"pid {pid} expected {v}, got {srp[pid]}"
    assert prp == {0: 0.75, 1: 1.0, 2: 1.0}, prp
    assert step_info_final["round_metrics_by_pid"][0]["round_payoff"] == 25.0
    assert step_info_final["prediction_metrics_by_pid"][0]["prediction_score"] == 0.75
    assert step_info_final["prediction_metrics_by_pid"][0]["prediction_exact_rate"] == 0.5
    assert step_info_final["prediction_metrics_by_pid"][0]["prediction_mae"] == 5.0


if __name__ == "__main__":
    test_pgg_predict()
    print("OK")
