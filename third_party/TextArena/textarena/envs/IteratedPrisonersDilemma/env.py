import re
from typing import Dict, Any, Optional, Tuple

import textarena as ta
# from textarena.envs.IteratedPrisonersDilemma.renderer import create_board_str  # add when you have a renderer


class IteratedPrisonersDilemmaEnv(ta.Env):
    def __init__(
        self,
        num_rounds: int = 5,
        communication_turns: int = 3,
        cooperate_reward: int = 3,
        defect_reward: int = 5,
        sucker_reward: int = 0,
        mutual_defect_reward: int = 1,
        # opt-in behavior — all default to legacy off so IteratedPrisonersDilemma-v0 is unchanged.
        enable_broadcast_comm: bool = False,   # simultaneous {msg} broadcast, PGG-style
        enable_prediction: bool = False,       # extra "prediction" phase between conversation and decision
        use_step_rewards: bool = False,        # write per-round payoff into state.step_info["step_rewards_by_pid"]
        prediction_reward: float = 1.0,        # awarded on exact match
    ):
        # game/round structure
        self.num_rounds = num_rounds
        self.conversation_rounds = communication_turns

        # payoff matrix (constant across rounds)
        self.cooperate_reward = cooperate_reward
        self.defect_reward = defect_reward
        self.sucker_reward = sucker_reward
        self.mutual_defect_reward = mutual_defect_reward

        # opt-in flags
        self.enable_broadcast_comm = enable_broadcast_comm
        self.enable_prediction = enable_prediction
        self.use_step_rewards = use_step_rewards
        self.prediction_reward = prediction_reward

        # One typed square-bracket grammar for every phase. UB may place this
        # payload inside its shared \boxed{...} answer envelope before it reaches
        # the environment.
        self.public_message_pattern = re.compile(
            r"^\s*\[Message\s*:\s*(.*?)\s*\]\s*$", re.IGNORECASE | re.DOTALL
        )
        self.prediction_pattern = re.compile(
            r"^\s*\[Prediction\s*:\s*(Cooperate|Defect)\s*\]\s*$", re.IGNORECASE
        )
        self.decision_pattern = re.compile(
            r"^\s*\[Action\s*:\s*(Cooperate|Defect)\s*\]\s*$", re.IGNORECASE
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
            "scores": {0: 0, 1: 0},
            # opt-in scratch space; unused unless the corresponding flag is set
            "pending_messages": {},
            "pending_predictions": {},
            "last_predictions": {},
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        base = (
            f"You are Player {player_id} in an Iterated Prisoner's Dilemma spanning "
            f"{game_state['num_rounds']} rounds.\n\n"
            f"Game Structure:\n"
            f"- Before each decision you have {game_state['total_conversation_rounds']} "
            f"turns to communicate.\n"
            f"- After that, both players simultaneously choose whether to cooperate or defect.\n\n"
            f"Payoff Matrix (fixed each round):\n"
            f"- Both Cooperate ➜ each {self.cooperate_reward}\n"
            f"- Both Defect ➜ each {self.mutual_defect_reward}\n"
            f"- One Defects, one Cooperates ➜ Defector {self.defect_reward}, "
            f"Cooperator {self.sucker_reward}\n\n"
        )
        if self.enable_broadcast_comm:
            comm_block = (
                f"How to Play:\n"
                f"- During conversation: send a public message using '[Message: your message]'. "
                f"Communication is SIMULTANEOUS: both players' messages are revealed only "
                f"after both submit. Only the message inside the command is broadcast.\n"
                f"  Example: [Message: Let's both cooperate this round.]\n"
            )
        else:
            comm_block = (
                f"How to Play:\n"
                f"- During conversation: type any text you wish.\n"
            )
        pred_block = ""
        if self.enable_prediction:
            pred_block = (
                f"- During prediction phase: privately predict your opponent's next decision as "
                f"'[Prediction: Cooperate]' or '[Prediction: Defect]'. A correct "
                f"prediction earns +{self.prediction_reward}. This is private to you.\n"
            )
        decision_block = (
            f"- During decision phase: submit exactly '[Action: Cooperate]' or "
            f"'[Action: Defect]' (case-insensitive).\n\n"
            "The payoff matrix will remain the same every round:\n"
            f"- Both Cooperate: {self.cooperate_reward}\n"
            f"- Both Defect: {self.mutual_defect_reward}\n"
            f"- If you Defect while the other Cooperates: {self.defect_reward}\n"
            f"- If you Cooperate while the other Defects: {self.sucker_reward}"
        )
        return base + comm_block + pred_block + decision_block + "\n\n" + self._phase_instruction("conversation")

    def _phase_instruction(self, phase: Optional[str] = None) -> str:
        """Return a final, unambiguous instruction for the current query."""
        phase = phase or self.state.game_state["phase"]
        round_number = self.state.game_state["round"]
        if phase == "conversation":
            if self.enable_broadcast_comm:
                requirement = "Reply with exactly one command: [Message: your public message]"
            else:
                requirement = "Reply with the message you want to send."
        elif phase == "prediction":
            requirement = "Reply with exactly one command: [Prediction: Cooperate] or [Prediction: Defect]"
        elif phase == "decision":
            requirement = "Reply with exactly one command: [Action: Cooperate] or [Action: Defect]"
        else:
            raise ValueError(f"Unknown IPD phase: {phase}")
        return f"CURRENT PHASE: {phase.upper()} (round {round_number}).\nREQUIRED OUTPUT: {requirement}"

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(
            to_id=self.state.current_player_id,
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        match self.state.game_state["phase"]:
            case "conversation": self._handle_conversation_phase(action)
            case "prediction":   self._handle_prediction_phase(action)
            case "decision":     self._handle_decision_phase(action)
        return self.state.step()

    # ---- conversation ---------------------------------------------------

    def _handle_conversation_phase(self, action: str):
        if self.enable_broadcast_comm:
            self._handle_conversation_broadcast(action)
        else:
            self._handle_conversation_legacy(action)

    def _handle_conversation_legacy(self, action: str):
        # legacy alternating conversation — opponent sees each utterance immediately
        self.state.add_observation(
            to_id=1 - self.state.current_player_id,
            from_id=self.state.current_player_id,
            message=action.strip(),
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        if self.state.current_player_id == 1:
            self.state.game_state["conversation_round"] += 1
            if self.state.game_state["conversation_round"] >= self.state.game_state["total_conversation_rounds"]:
                self._exit_conversation_phase()

    def _extract_public_message(self, action: str) -> Optional[str]:
        match = self.public_message_pattern.fullmatch(action)
        if match and match.group(1).strip():
            return match.group(1).strip()
        return None

    def _handle_conversation_broadcast(self, action: str):
        # PGG-style: buffer per pid, broadcast only after both submit
        public_msg = self._extract_public_message(action)
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[self.state.current_player_id] = public_msg is not None
        self.state.game_state["pending_messages"][self.state.current_player_id] = public_msg
        if all(p in self.state.game_state["pending_messages"] for p in range(2)):
            lines = []
            for sender_id in range(2):
                m = self.state.game_state["pending_messages"][sender_id]
                lines.append(f"Player {sender_id}: {m}" if m else f"Player {sender_id}: [remained silent]")
            full = "Messages from this turn:\n" + "\n".join(lines)
            for receiver_id in range(2):
                self.state.add_observation(
                    to_id=receiver_id,
                    from_id=ta.GAME_ID,
                    message=full,
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            self.state.game_state["pending_messages"] = {}
            self.state.game_state["conversation_round"] += 1
            if self.state.game_state["conversation_round"] >= self.state.game_state["total_conversation_rounds"]:
                self._exit_conversation_phase()

    def _exit_conversation_phase(self):
        # transition to prediction (if enabled) or decision
        if self.enable_prediction:
            self.state.game_state["phase"] = "prediction"
            self.state.game_state["pending_predictions"] = {}
            # private prompt to each player
            for pid in range(2):
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message=(
                        f"Prediction phase for round {self.state.game_state['round']} is private. "
                        f"Correct predictions earn +{self.prediction_reward}.\n"
                        f"{self._phase_instruction('prediction')}"
                    ),
                    observation_type=ta.ObservationType.GAME_BOARD,
                )
        else:
            self.state.game_state["phase"] = "decision"
            self.state.add_observation(
                message=(
                    f"Conversation finished for round {self.state.game_state['round']}.\n"
                    f"{self._phase_instruction('decision')}"
                ),
                observation_type=ta.ObservationType.GAME_BOARD,
            )

    # ---- prediction ------------------------------------------------------

    def _parse_prediction(self, action: str) -> Optional[str]:
        match = self.prediction_pattern.fullmatch(action)
        return match.group(1).lower() if match else None

    def _handle_prediction_phase(self, action: str):
        pred = self._parse_prediction(action)
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[self.state.current_player_id] = pred is not None
        # store None if unparseable; scored as a miss
        self.state.game_state["pending_predictions"][self.state.current_player_id] = pred
        if all(p in self.state.game_state["pending_predictions"] for p in range(2)):
            self.state.game_state["last_predictions"] = dict(self.state.game_state["pending_predictions"])
            self.state.game_state["pending_predictions"] = {}
            self.state.game_state["phase"] = "decision"
            self.state.add_observation(
                message=(
                    f"Predictions are locked in.\n{self._phase_instruction('decision')}"
                ),
                observation_type=ta.ObservationType.GAME_BOARD,
            )

    # ---- decision --------------------------------------------------------

    def _handle_decision_phase(self, action: str):
        match = self.decision_pattern.fullmatch(action)
        if match is None:
            self.state.step_info.setdefault("phase_format_valid_by_pid", {})[self.state.current_player_id] = False
            self.state.set_invalid_move(
                reason="Decision must be exactly '[Action: Cooperate]' or '[Action: Defect]'."
            )
            return
        decision = match.group(1).lower()
        self.state.step_info.setdefault("phase_format_valid_by_pid", {})[self.state.current_player_id] = True
        self.state.game_state["decisions"][self.state.current_player_id] = decision

        if all(d is not None for d in self.state.game_state["decisions"].values()):
            self._resolve_round()
            self.state.game_state["round"] += 1
            if self.state.game_state["round"] > self.state.game_state["num_rounds"]:
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
                self.state.add_observation(
                    message=(
                        f"--- Starting Round {self.state.game_state['round']} ---\n"
                        f"{self._phase_instruction('conversation')}"
                    ),
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )

    # ---- resolution ------------------------------------------------------

    def _resolve_round(self):
        d0 = self.state.game_state["decisions"][0]
        d1 = self.state.game_state["decisions"][1]

        if d0 == d1 == "cooperate":                 r0 = r1 = self.cooperate_reward;                    outcome = "Both players cooperated."
        elif d0 == d1 == "defect":                  r0 = r1 = self.mutual_defect_reward;                outcome = "Both players defected."
        elif d0 == "cooperate" and d1 == "defect":  r0, r1 = self.sucker_reward, self.defect_reward;    outcome = "Player 0 cooperated, Player 1 defected."
        else:                                       r0, r1 = self.defect_reward, self.sucker_reward;    outcome = "Player 0 defected, Player 1 cooperated."

        self.state.game_state["scores"][0] += r0
        self.state.game_state["scores"][1] += r1
        self.state.step_info["round_decisions_by_pid"] = {0: d0, 1: d1}
        self.state.step_info["mutual_cooperation"] = d0 == d1 == "cooperate"

        # prediction scoring — reward is added on top of payoff for the same step
        pred_bonus = {0: 0.0, 1: 0.0}
        pred_msgs = {0: "", 1: ""}
        if self.enable_prediction:
            actual = {0: d0, 1: d1}
            for pid in range(2):
                predicted = self.state.game_state["last_predictions"].get(pid)
                truth = actual[1 - pid]
                if predicted is not None and predicted == truth:
                    pred_bonus[pid] = float(self.prediction_reward)
                    pred_msgs[pid] = f"Your prediction '{predicted}' was CORRECT (+{self.prediction_reward})."
                elif predicted is None:
                    pred_msgs[pid] = f"Your prediction was unparseable; opponent chose '{truth}'."
                else:
                    pred_msgs[pid] = f"Your prediction '{predicted}' was WRONG; opponent chose '{truth}'."
                self.state.add_observation(
                    to_id=pid,
                    from_id=ta.GAME_ID,
                    message=pred_msgs[pid],
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )

        # publish per-step env reward via step_info; collector distributes to each pid's last step
        if self.use_step_rewards:
            step_rewards = self.state.step_info.setdefault("step_rewards_by_pid", {})
            step_rewards[0] = float(r0)
            step_rewards[1] = float(r1)
            if self.enable_prediction:
                self.state.step_info["prediction_rewards_by_pid"] = pred_bonus

        self.state.add_observation(
            message=(
                f"Round {self.state.game_state['round']} results:\n{outcome}\n"
                f"Player 0 earned {r0} (total {self.state.game_state['scores'][0]}), "
                f"Player 1 earned {r1} (total {self.state.game_state['scores'][1]})."
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

    def _determine_winner(self):
        s0 = self.state.game_state["scores"][0]
        s1 = self.state.game_state["scores"][1]
        if s0 == s1:
            self.state.set_draw(reason=f"Draw! Both players scored {s0}.")
        else:
            winner = 0 if s0 > s1 else 1
            self.state.set_winner(player_id=winner, reason=f"Player {winner} wins {max(s0, s1)} - {min(s0, s1)}.")
