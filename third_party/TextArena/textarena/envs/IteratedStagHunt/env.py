import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import textarena as ta


class IteratedStagHuntEnv(ta.Env):
    def __init__(
        self,
        num_rounds: int = 5,
        conversation_rounds: int = 3,
        mutual_stag_reward: int = 10,
        single_hare_reward: int = 8,
        single_stag_reward: int = 1,
        mutual_hare_reward: int = 5,
        randomize_payoff: bool = False,
        enable_broadcast_comm: bool = False,
        enable_prediction: bool = False,
        use_step_rewards: bool = False,
        prediction_reward: float = 1.0,
        typed_actions: bool = False,
    ):
        self.num_rounds = num_rounds
        self.conversation_rounds = conversation_rounds
        self.mutual_stag_reward = mutual_stag_reward
        self.single_hare_reward = single_hare_reward
        self.single_stag_reward = single_stag_reward
        self.mutual_hare_reward = mutual_hare_reward
        self.randomize_payoff = randomize_payoff
        self.enable_broadcast_comm = enable_broadcast_comm
        self.enable_prediction = enable_prediction
        self.use_step_rewards = use_step_rewards
        self.prediction_reward = prediction_reward
        self.typed_actions = typed_actions

        self.stag_pattern = re.compile(r"\[Stag\]", re.IGNORECASE)
        self.message_pattern = re.compile(
            r"^\s*\[Message\s*:\s*(.*?)\s*\]\s*$", re.IGNORECASE | re.DOTALL
        )
        self.prediction_pattern = re.compile(
            r"^\s*\[Prediction\s*:\s*(Stag|Hare)\s*\]\s*$", re.IGNORECASE
        )
        self.action_pattern = re.compile(
            r"^\s*\[Action\s*:\s*(Stag|Hare)\s*\]\s*$", re.IGNORECASE
        )

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {
            "round": 1,
            "num_rounds": self.num_rounds,
            "phase": "conversation",
            "conversation_round": 0,
            "total_conversation_rounds": self.conversation_rounds,
            "decisions": {0: None, 1: None},
            "total_payoff": {0: 0, 1: 0},
            "pending_messages": {},
            "pending_predictions": {},
            "last_predictions": {},
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._create_round_payoff_matrix()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        communication = (
            "Messages are revealed simultaneously after both players submit."
            if self.enable_broadcast_comm
            else "Messages are shown to the opponent immediately."
        )
        prediction = ""
        if self.enable_prediction:
            prediction = (
                f"- Privately predict the opponent's choice; a correct prediction earns "
                f"+{self.prediction_reward}.\n"
            )
        return (
            f"You are Player {player_id} in a {game_state['num_rounds']}-round Iterated Stag Hunt.\n\n"
            f"Each round has {game_state['total_conversation_rounds']} communication turns, "
            f"then simultaneous decisions. {communication}\n"
            f"{prediction}"
            "Payoffs are announced at the start of every round.\n"
            "Your goal is to maximize cumulative payoff.\n\n"
            + self._phase_instruction("conversation")
        )

    def _phase_instruction(self, phase: Optional[str] = None) -> str:
        phase = phase or self.state.game_state["phase"]
        round_number = self.state.game_state["round"]
        if self.typed_actions:
            required = {
                "conversation": "Reply with exactly one command: [Message: your public message]",
                "prediction": "Reply with exactly one command: [Prediction: Stag] or [Prediction: Hare]",
                "decision": "Reply with exactly one command: [Action: Stag] or [Action: Hare]",
            }[phase]
        else:
            required = (
                "Type the message you want to send."
                if phase == "conversation"
                else "Reply with [Stag] or [Hare]."
            )
        return f"CURRENT PHASE: {phase.upper()} (round {round_number}).\nREQUIRED OUTPUT: {required}"

    def _create_round_payoff_matrix(self) -> None:
        if not self.randomize_payoff:
            self.mutual_stag_payoff = self.mutual_stag_reward
            self.single_stag_payoff = self.single_stag_reward
            self.single_hare_payoff = self.single_hare_reward
            self.mutual_hare_payoff = self.mutual_hare_reward
        else:
            self.single_stag_payoff = self.single_stag_reward
            self.mutual_hare_payoff = np.random.randint(
                self.single_stag_payoff + 1, self.mutual_hare_reward + 1
            )
            self.single_hare_payoff = np.random.randint(
                self.mutual_hare_payoff, self.single_hare_reward + 1
            )
            self.mutual_stag_payoff = np.random.randint(
                self.single_hare_payoff + 1, self.mutual_stag_reward + 1
            )
        self.state.add_observation(
            message=(
                f"Starting Round {self.state.game_state['round']} with payoff matrix:\n"
                f"- Both Stag: {self.mutual_stag_payoff} each\n"
                f"- Both Hare: {self.mutual_hare_payoff} each\n"
                f"- Split: Hare gets {self.single_hare_payoff}; Stag gets {self.single_stag_payoff}\n"
                f"{self._phase_instruction('conversation')}"
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        cid = self.state.current_player_id
        self.state.add_observation(
            to_id=cid,
            from_id=cid,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        phase = self.state.game_state["phase"]
        if phase == "conversation":
            self._handle_conversation_phase(action)
        elif phase == "prediction":
            self._handle_prediction_phase(action)
        else:
            self._handle_decision_phase(action)
        return self.state.step()

    def _handle_conversation_phase(self, action: str):
        cid = self.state.current_player_id
        if not self.enable_broadcast_comm:
            self.state.add_observation(
                to_id=1 - cid,
                from_id=cid,
                message=action.strip(),
                observation_type=ta.ObservationType.PLAYER_ACTION,
            )
            if cid == 1:
                self.state.game_state["conversation_round"] += 1
                if self.state.game_state["conversation_round"] >= self.conversation_rounds:
                    self._exit_conversation()
            return

        match = self.message_pattern.fullmatch(action)
        message = match.group(1).strip() if match and match.group(1).strip() else None
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = message is not None
        self.state.game_state["pending_messages"][cid] = message
        if len(self.state.game_state["pending_messages"]) == 2:
            lines = [
                f"Player {pid}: {self.state.game_state['pending_messages'][pid]}"
                if self.state.game_state["pending_messages"][pid]
                else f"Player {pid}: [remained silent]"
                for pid in range(2)
            ]
            for pid in range(2):
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message="Messages from this turn:\n" + "\n".join(lines),
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            self.state.game_state["pending_messages"] = {}
            self.state.game_state["conversation_round"] += 1
            if self.state.game_state["conversation_round"] >= self.conversation_rounds:
                self._exit_conversation()

    def _exit_conversation(self):
        phase = "prediction" if self.enable_prediction else "decision"
        self.state.game_state["phase"] = phase
        for pid in range(2):
            self.state.add_observation(
                to_id=pid,
                from_id=ta.GAME_ID,
                message=self._phase_instruction(phase),
                observation_type=ta.ObservationType.GAME_BOARD,
            )

    def _handle_prediction_phase(self, action: str):
        cid = self.state.current_player_id
        match = self.prediction_pattern.fullmatch(action)
        prediction = match.group(1).lower() if match else None
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = prediction is not None
        self.state.game_state["pending_predictions"][cid] = prediction
        if len(self.state.game_state["pending_predictions"]) == 2:
            self.state.game_state["last_predictions"] = dict(
                self.state.game_state["pending_predictions"]
            )
            self.state.game_state["pending_predictions"] = {}
            self.state.game_state["phase"] = "decision"
            self.state.add_observation(
                message=self._phase_instruction("decision"),
                observation_type=ta.ObservationType.GAME_BOARD,
            )

    def _handle_decision_phase(self, action: str):
        cid = self.state.current_player_id
        if self.typed_actions:
            match = self.action_pattern.fullmatch(action)
            if match is None:
                self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = False
                self.state.set_invalid_move(
                    reason="Decision must be exactly '[Action: Stag]' or '[Action: Hare]'."
                )
                return
            decision = match.group(1).lower()
            self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = True
        else:
            decision = "stag" if self.stag_pattern.search(action) else "hare"
        self.state.game_state["decisions"][cid] = decision
        if all(value is not None for value in self.state.game_state["decisions"].values()):
            self._resolve_round()
            self.state.game_state["round"] += 1
            if self.state.game_state["round"] > self.num_rounds:
                self._determine_winner()
            else:
                self.state.game_state.update({
                    "phase": "conversation",
                    "conversation_round": 0,
                    "decisions": {0: None, 1: None},
                    "pending_messages": {},
                    "pending_predictions": {},
                    "last_predictions": {},
                })
                self._create_round_payoff_matrix()

    def _resolve_round(self):
        gs = self.state.game_state
        d0, d1 = gs["decisions"][0], gs["decisions"][1]
        if d0 == d1:
            payoff = self.mutual_stag_payoff if d0 == "stag" else self.mutual_hare_payoff
            round_payoffs = {0: payoff, 1: payoff}
        else:
            round_payoffs = {
                pid: self.single_stag_payoff if gs["decisions"][pid] == "stag" else self.single_hare_payoff
                for pid in range(2)
            }
        for pid, payoff in round_payoffs.items():
            gs["total_payoff"][pid] += payoff

        if self.use_step_rewards:
            self.state.step_info["step_rewards_by_pid"] = {
                pid: float(payoff) for pid, payoff in round_payoffs.items()
            }
        self.state.step_info["round_metrics_by_pid"] = {
            pid: {
                "round_payoff": float(round_payoffs[pid]),
                "cumulative_score": float(gs["total_payoff"][pid]),
                "decision": gs["decisions"][pid],
            }
            for pid in range(2)
        }
        self.state.step_info["round_decisions_by_pid"] = dict(gs["decisions"])
        self.state.step_info["mutual_cooperation"] = d0 == d1 == "stag"

        if self.enable_prediction:
            rewards, metrics = {}, {}
            for pid in range(2):
                predicted = gs["last_predictions"].get(pid)
                truth = gs["decisions"][1 - pid]
                score = float(predicted == truth)
                rewards[pid] = self.prediction_reward * score
                metrics[pid] = {
                    "prediction_score": score,
                    "prediction_exact_rate": score,
                }
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message=f"Opponent chose {truth}; prediction score {score:.1f}.",
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            if self.use_step_rewards:
                self.state.step_info["prediction_rewards_by_pid"] = rewards
            self.state.step_info["prediction_metrics_by_pid"] = metrics

        self.state.add_observation(
            message=(
                f"Round {gs['round']} results:\n"
                + "\n".join(
                    f"Player {pid} chose {gs['decisions'][pid]}, earned {round_payoffs[pid]}, "
                    f"total {gs['total_payoff'][pid]}."
                    for pid in range(2)
                )
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

    def _determine_winner(self):
        scores = self.state.game_state["total_payoff"]
        if scores[0] == scores[1]:
            self.state.set_draw(reason=f"Tie: Player 0={scores[0]}, Player 1={scores[1]}.")
        else:
            winner = 0 if scores[0] > scores[1] else 1
            self.state.set_winner(
                player_id=winner,
                reason=f"Player {winner} wins. Final scores: P0={scores[0]}, P1={scores[1]}.",
            )
