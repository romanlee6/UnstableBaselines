import itertools
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class ThreePlayerIPDEnv(ta.Env):
    def __init__(
        self,
        num_rounds: int = 5,
        communication_turns: int = 3,
        cooperate_reward: int = 3,
        defect_reward: int = 5,
        sucker_reward: int = 0,
        mutual_defect_reward: int = 1,
        enable_broadcast_comm: bool = False,
        enable_prediction: bool = False,
        use_step_rewards: bool = False,
        prediction_reward: float = 1.0,
        typed_actions: bool = False,
    ):
        self.num_rounds = num_rounds
        self.conversation_rounds = communication_turns
        self.R, self.T, self.S, self.P = (
            cooperate_reward,
            defect_reward,
            sucker_reward,
            mutual_defect_reward,
        )
        self.enable_broadcast_comm = enable_broadcast_comm
        self.enable_prediction = enable_prediction
        self.use_step_rewards = use_step_rewards
        self.prediction_reward = prediction_reward
        self.typed_actions = typed_actions
        self.token_pat = re.compile(r"\[\s*(\d+)\s+(cooperate|defect)\s*\]", re.I)
        self.message_pattern = re.compile(
            r"^\s*\[Message\s*:\s*(.*?)\s*\]\s*$", re.I | re.S
        )
        self.prediction_pattern = re.compile(
            r"^\s*\[Prediction\s*:\s*(.*?)\s*\]\s*$", re.I | re.S
        )
        self.action_pattern = re.compile(
            r"^\s*\[Action\s*:\s*(.*?)\s*\]\s*$", re.I | re.S
        )
        self.entry_pattern = re.compile(r"P(\d+)\s*=\s*(Cooperate|Defect)", re.I)

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert num_players == 3, "ThreePlayerIPD requires exactly three players."
        self.state = ta.FFAMultiPlayerState(num_players=num_players, seed=seed)
        game_state = {
            "round": 1,
            "num_rounds": self.num_rounds,
            "phase": "conversation",
            "conversation_round": 0,
            "total_conversation_rounds": self.conversation_rounds,
            "decisions": self._empty_target_map(),
            "scores": {p: 0 for p in range(num_players)},
            "acted": {p: False for p in range(num_players)},
            "pending_messages": {},
            "pending_predictions": {},
            "last_predictions": {},
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._announce_round()

    def _empty_target_map(self):
        return {
            p: {q: None for q in range(3) if q != p}
            for p in range(3)
        }

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        chat = (
            "Messages are revealed simultaneously after all players submit."
            if self.enable_broadcast_comm
            else "Messages are shown to the other players immediately."
        )
        prediction = ""
        if self.enable_prediction:
            prediction = (
                "- After chat, privately predict each opponent's action toward you. "
                f"The averaged exact-match score earns up to +{self.prediction_reward}.\n"
            )
        return (
            f"You are Player {player_id} in a three-player Iterated Prisoner's Dilemma "
            f"lasting {game_state['num_rounds']} rounds.\n"
            f"Each round has {game_state['total_conversation_rounds']} chat turns. {chat}\n"
            f"{prediction}"
            "You independently cooperate or defect against each opponent. Pairwise payoffs:\n"
            f"- cooperate/cooperate: {self.R} each\n"
            f"- defect/defect: {self.P} each\n"
            f"- defector/cooperator: {self.T}/{self.S}\n"
            "Your round payoff is the sum of both pairwise payoffs.\n\n"
            + self._phase_instruction(player_id, "conversation")
        )

    def _phase_instruction(self, player_id: int, phase: Optional[str] = None) -> str:
        phase = phase or self.state.game_state["phase"]
        opponents = [pid for pid in range(3) if pid != player_id]
        if self.typed_actions:
            example = ", ".join(f"P{pid}=Cooperate" for pid in opponents)
            required = {
                "conversation": "Reply with exactly one command: [Message: your public message]",
                "prediction": f"Reply with exactly one command; use Cooperate or Defect for every opponent: [Prediction: {example}]",
                "decision": f"Reply with exactly one command; use Cooperate or Defect for every opponent: [Action: {example}]",
            }[phase]
        else:
            required = (
                "Type the message you want to send."
                if phase == "conversation"
                else "Submit one '[pid cooperate]' or '[pid defect]' token per opponent."
            )
        return (
            f"CURRENT PHASE: {phase.upper()} (round {self.state.game_state['round']}).\n"
            f"REQUIRED OUTPUT: {required}"
        )

    def _announce_round(self):
        for pid in range(3):
            self.state.add_observation(
                to_id=pid,
                from_id=ta.GAME_ID,
                message=(
                    f"--- Starting Round {self.state.game_state['round']} ---\n"
                    f"{self._phase_instruction(pid, 'conversation')}"
                ),
                observation_type=ta.ObservationType.GAME_MESSAGE,
            )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        cid = self.state.current_player_id
        self.state.add_observation(
            from_id=cid,
            to_id=cid,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        phase = self.state.game_state["phase"]
        if phase == "conversation":
            self._conversation_phase(action)
        elif phase == "prediction":
            self._prediction_phase(action)
        else:
            self._decision_phase(action)
        return self.state.step()

    @staticmethod
    def _clean_message(msg: str) -> str:
        return re.sub(r"\s+", " ", msg).strip()

    def _conversation_phase(self, msg: str):
        cid = self.state.current_player_id
        if not self.enable_broadcast_comm:
            for pid in range(3):
                if pid != cid:
                    self.state.add_observation(
                        from_id=cid,
                        to_id=pid,
                        message=self._clean_message(msg),
                        observation_type=ta.ObservationType.PLAYER_ACTION,
                    )
            if cid == 2:
                self.state.game_state["conversation_round"] += 1
                if self.state.game_state["conversation_round"] >= self.conversation_rounds:
                    self._exit_conversation()
            return

        match = self.message_pattern.fullmatch(msg)
        public = match.group(1).strip() if match and match.group(1).strip() else None
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = public is not None
        self.state.game_state["pending_messages"][cid] = public
        if len(self.state.game_state["pending_messages"]) == 3:
            lines = [
                f"Player {pid}: {self.state.game_state['pending_messages'][pid]}"
                if self.state.game_state["pending_messages"][pid]
                else f"Player {pid}: [remained silent]"
                for pid in range(3)
            ]
            for pid in range(3):
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
        for pid in range(3):
            self.state.add_observation(
                to_id=pid,
                from_id=ta.GAME_ID,
                message=self._phase_instruction(pid, phase),
                observation_type=ta.ObservationType.GAME_BOARD,
            )

    def _parse_target_command(self, action: str, pattern, cid: int):
        match = pattern.fullmatch(action)
        entries = self.entry_pattern.findall(match.group(1)) if match else []
        parsed = {int(pid): choice.lower() for pid, choice in entries}
        expected = {pid for pid in range(3) if pid != cid}
        valid = match is not None and len(entries) == len(expected) and set(parsed) == expected
        return parsed if valid else None

    def _prediction_phase(self, action: str):
        cid = self.state.current_player_id
        prediction = self._parse_target_command(action, self.prediction_pattern, cid)
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = prediction is not None
        self.state.game_state["pending_predictions"][cid] = prediction or {}
        if len(self.state.game_state["pending_predictions"]) == 3:
            self.state.game_state["last_predictions"] = dict(
                self.state.game_state["pending_predictions"]
            )
            self.state.game_state["pending_predictions"] = {}
            self.state.game_state["phase"] = "decision"
            for pid in range(3):
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message=self._phase_instruction(pid, "decision"),
                    observation_type=ta.ObservationType.GAME_BOARD,
                )

    def _decision_phase(self, action: str):
        cid = self.state.current_player_id
        gs = self.state.game_state
        if self.typed_actions:
            decision = self._parse_target_command(action, self.action_pattern, cid)
            if decision is None:
                self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = False
                self.state.set_invalid_move(
                    reason="Decision must contain exactly one valid action for each opponent."
                )
                return
            self.state.step_info.setdefault("phase_format_valid_by_pid", {})[cid] = True
            gs["decisions"][cid] = decision
        else:
            for pid_str, choice in self.token_pat.findall(action):
                target = int(pid_str)
                if target in gs["decisions"][cid]:
                    gs["decisions"][cid][target] = choice.lower()
            for target, choice in gs["decisions"][cid].items():
                if choice is None:
                    gs["decisions"][cid][target] = "cooperate"
        gs["acted"][cid] = True
        if all(gs["acted"].values()):
            self._resolve_round()
            gs["round"] += 1
            if gs["round"] > gs["num_rounds"]:
                self._end_game()
            else:
                gs.update({
                    "phase": "conversation",
                    "conversation_round": 0,
                    "acted": {p: False for p in range(3)},
                    "decisions": self._empty_target_map(),
                    "pending_messages": {},
                    "pending_predictions": {},
                    "last_predictions": {},
                })
                self._announce_round()

    def _pair_payoff(self, a: str, b: str) -> Tuple[int, int]:
        if a == b == "cooperate":
            return self.R, self.R
        if a == b == "defect":
            return self.P, self.P
        return (self.S, self.T) if a == "cooperate" else (self.T, self.S)

    def _resolve_round(self):
        gs = self.state.game_state
        decisions = gs["decisions"]
        gains = {pid: 0 for pid in range(3)}
        lines = []
        for i, j in itertools.combinations(range(3), 2):
            pi, pj = self._pair_payoff(decisions[i][j], decisions[j][i])
            gains[i] += pi
            gains[j] += pj
            lines.append(
                f"P{i}→P{j}: {decisions[i][j]}, P{j}→P{i}: {decisions[j][i]} "
                f"(payoffs {pi}/{pj})"
            )
        for pid, gain in gains.items():
            gs["scores"][pid] += gain

        if self.use_step_rewards:
            self.state.step_info["step_rewards_by_pid"] = {
                pid: float(gain) for pid, gain in gains.items()
            }
        self.state.step_info["round_metrics_by_pid"] = {
            pid: {
                "round_payoff": float(gains[pid]),
                "cumulative_score": float(gs["scores"][pid]),
            }
            for pid in range(3)
        }
        self.state.step_info["round_decisions_by_pid"] = {
            pid: dict(targets) for pid, targets in decisions.items()
        }

        if self.enable_prediction:
            prediction_rewards, prediction_metrics = {}, {}
            for pid in range(3):
                opponents = [other for other in range(3) if other != pid]
                predicted = gs["last_predictions"].get(pid, {})
                matches = [
                    predicted.get(other) == decisions[other][pid]
                    for other in opponents
                ]
                score = sum(matches) / len(matches)
                prediction_rewards[pid] = self.prediction_reward * score
                prediction_metrics[pid] = {
                    "prediction_score": score,
                    "prediction_exact_rate": score,
                }
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message=f"Opponent-action prediction score {score:.3f}.",
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            if self.use_step_rewards:
                self.state.step_info["prediction_rewards_by_pid"] = prediction_rewards
            self.state.step_info["prediction_metrics_by_pid"] = prediction_metrics

        self.state.add_observation(
            message=(
                f"Round {gs['round']} results:\n"
                + "\n".join(lines)
                + "\n"
                + "; ".join(
                    f"P{pid} gained {gains[pid]} (total {gs['scores'][pid]})"
                    for pid in range(3)
                )
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

    def _end_game(self):
        scores = self.state.game_state["scores"]
        ranked = sorted(scores, key=lambda pid: (scores[pid], -pid))
        groups = []
        for pid in ranked:
            if not groups or scores[pid] != scores[groups[-1][0]]:
                groups.append([pid])
            else:
                groups[-1].append(pid)
        if len(groups) == 1:
            rewards = {pid: 0.0 for pid in groups[0]}
        else:
            rewards = {
                pid: -1.0 + 2.0 * group_index / (len(groups) - 1)
                for group_index, group in enumerate(groups)
                for pid in group
            }
        self.state.set_game_outcome(
            reward_dict=rewards,
            reason="Final scores: " + ", ".join(f"P{pid}={scores[pid]}" for pid in range(3)),
        )
