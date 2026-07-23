import re
from typing import Tuple, Dict, Callable


def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"
TEMPLATE_PARTS = {
    "default": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{{}}.\nObservation: {obs}\n"
    },
    "qwen3-zs": {
        "user": lambda obs: f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-sp": {
        "user": lambda obs:  f"<|im_start|>user\nYou are playing a single-player game. Make valid actions to solve it completely.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-reasoning": {
        "user": lambda obs: f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n<think>"
    },
    "gemma3-zs": {
        "user": lambda obs: f"<bos><start_of_turn>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n"
    },
    "llama-instruct-zs": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are playing a two-player zero-sum game. Make valid actions to win.<|eot_id|>",
        "user": lambda obs: f"<|start_header_id|>user<|end_header_id|>\n\nCurrent Observation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>"
    },
}
def apply_template(template_name: str, observation: str) -> str:
    parts = TEMPLATE_PARTS.get(template_name)
    return format_template(system=parts.get("system", ""), user=parts["user"](observation), assistant=parts.get("assistant", ""))


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    """Extract the final ``\\boxed{...}`` payload without changing its syntax.

    Game phases use distinct payload delimiters: ``{message}`` for communication,
    ``<prediction>`` for prediction, and ``[action]`` for decisions.  The old
    regex stopped at the first closing brace and then added square brackets,
    corrupting a valid communication payload such as ``\\boxed{{hello}}``.
    """
    payloads = []
    marker = r"\boxed{"
    start = 0
    while (box_start := raw_action.find(marker, start)) != -1:
        payload_start = box_start + len(marker)
        depth, pos = 1, payload_start
        while pos < len(raw_action) and depth:
            if raw_action[pos] == "{": depth += 1
            elif raw_action[pos] == "}": depth -= 1
            pos += 1
        if depth == 0:
            payloads.append(raw_action[payload_start:pos - 1].strip())
            start = pos
        else:
            start = payload_start

    payload = payloads[-1] if payloads else ""
    # Preserve phase delimiters exactly.  Keep the historical bracket fallback
    # for legacy games whose boxed payload is an un-delimited action string.
    action = payload if payload[:1] in {"{", "<", "["} else (f"[{payload}]" if payload else raw_action)
    format_feedback = {"correct_answer_format": bool(payload)}
    return action, format_feedback


OBSERVATION_FORMATTING: Dict[str, Callable[[str], str]] = {key: (lambda key=key: lambda observation: apply_template(key, observation))() for key in TEMPLATE_PARTS}
ACTION_EXTRACTION = {"default": extract_action_and_format_feedback}
