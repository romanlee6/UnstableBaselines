"""Enhanced Stag Hunt and ThreePlayerIPD environment checks."""

import textarena as ta


def _pending_text(env, pid):
    return "\n".join(message for _, message, _ in env.state.observations[pid])


def test_stag_hunt_buffered_prediction_and_payoff():
    env = ta.make("IteratedStagHunt-Predict-v0")
    env.reset(num_players=2, seed=0)

    for chat_round in range(3):
        pid, _ = env.get_observation()
        assert pid == 0
        env.step(f"[Message: p0-r{chat_round}]")
        assert f"p0-r{chat_round}" not in _pending_text(env, 1)
        pid, _ = env.get_observation()
        assert pid == 1
        env.step(f"[Message: p1-r{chat_round}]")

    assert env.state.game_state["phase"] == "prediction"
    env.step("[Prediction: Stag]")
    env.step("[Prediction: Hare]")
    assert env.state.game_state["phase"] == "decision"
    env.step("[Action: Stag]")
    _, info = env.step("[Action: Stag]")

    assert info["step_rewards_by_pid"] == {0: 10.0, 1: 10.0}
    assert info["prediction_rewards_by_pid"] == {0: 1.0, 1: 0.0}
    assert info["round_metrics_by_pid"][0]["round_payoff"] == 10.0
    assert info["prediction_metrics_by_pid"][1]["prediction_exact_rate"] == 0.0


def test_three_player_ipd_buffered_prediction_and_pairwise_payoff():
    env = ta.make("ThreePlayerIPD-Predict-v0")
    env.reset(num_players=3, seed=0)

    env.step("[Message: p0]")
    assert "p0" not in _pending_text(env, 1)
    env.step("[Message: p1]")
    assert "p1" not in _pending_text(env, 2)
    env.step("[Message: p2]")
    assert all("p0" in _pending_text(env, pid) for pid in range(3))
    assert env.state.game_state["phase"] == "prediction"

    env.step("[Prediction: P1=Cooperate, P2=Cooperate]")
    env.step("[Prediction: P0=Cooperate, P2=Defect]")
    env.step("[Prediction: P0=Defect, P1=Defect]")
    assert env.state.game_state["phase"] == "decision"

    env.step("[Action: P1=Cooperate, P2=Defect]")
    env.step("[Action: P0=Cooperate, P2=Defect]")
    _, info = env.step("[Action: P0=Cooperate, P1=Defect]")

    assert info["step_rewards_by_pid"] == {0: 8.0, 1: 4.0, 2: 1.0}
    assert info["prediction_rewards_by_pid"] == {0: 1.0, 1: 1.0, 2: 1.0}
    assert info["round_metrics_by_pid"][0]["round_payoff"] == 8.0
    assert info["prediction_metrics_by_pid"][2]["prediction_score"] == 1.0


def test_legacy_variants_keep_immediate_chat_and_terminal_only_rewards():
    stag = ta.make("IteratedStagHunt-v0")
    stag.reset(num_players=2, seed=0)
    stag.step("legacy-stag-message")
    assert "legacy-stag-message" in _pending_text(stag, 1)

    three = ta.make("ThreePlayerIPD-v0")
    three.reset(num_players=3, seed=0)
    three.step("legacy-three-player-message")
    assert "legacy-three-player-message" in _pending_text(three, 1)
    assert "step_rewards_by_pid" not in three.state.step_info


def test_broadcast_controls_skip_prediction_but_keep_round_payoff():
    stag = ta.make("IteratedStagHunt-Broadcast-v0")
    stag.reset(num_players=2, seed=0)
    for _ in range(3):
        stag.step("[Message: coordinate]")
        stag.step("[Message: coordinate]")
    assert stag.state.game_state["phase"] == "decision"
    stag.step("[Action: Hare]")
    _, stag_info = stag.step("[Action: Hare]")
    assert stag_info["step_rewards_by_pid"] == {0: 5.0, 1: 5.0}
    assert "prediction_rewards_by_pid" not in stag_info

    three = ta.make("ThreePlayerIPD-Broadcast-v0")
    three.reset(num_players=3, seed=0)
    for pid in range(3):
        three.step(f"[Message: p{pid}]")
    assert three.state.game_state["phase"] == "decision"
    three.step("[Action: P1=Cooperate, P2=Cooperate]")
    three.step("[Action: P0=Cooperate, P2=Cooperate]")
    _, three_info = three.step("[Action: P0=Cooperate, P1=Cooperate]")
    assert three_info["step_rewards_by_pid"] == {0: 6.0, 1: 6.0, 2: 6.0}
    assert "prediction_rewards_by_pid" not in three_info

    public_goods = ta.make("PublicGoodsGame-Broadcast-v0")
    public_goods.reset(num_players=3, seed=0)
    for _ in range(3):
        for pid in range(3):
            public_goods.step(f"[Message: p{pid}]")
    assert public_goods.state.game_state["phase"] == "decision"
    public_goods.step("[Action: 10]")
    public_goods.step("[Action: 10]")
    _, pgg_info = public_goods.step("[Action: 10]")
    assert pgg_info["step_rewards_by_pid"] == {0: 25.0, 1: 25.0, 2: 25.0}
    assert "prediction_rewards_by_pid" not in pgg_info


if __name__ == "__main__":
    test_stag_hunt_buffered_prediction_and_payoff()
    test_three_player_ipd_buffered_prediction_and_pairwise_payoff()
    test_legacy_variants_keep_immediate_chat_and_terminal_only_rewards()
    test_broadcast_controls_skip_prediction_but_keep_round_payoff()
    print("OK")
