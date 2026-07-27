from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CompletionPreservingBatch:
    encoding: object
    prompt_lengths: List[int]
    sequence_lengths: List[int]
    original_lengths: List[int]
    prompt_tokens_dropped: List[int]


def _retain_prompt_tokens(
    token_ids: List[int], max_prompt_tokens: int, prefix_tokens: int
) -> tuple[List[int], int]:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    dropped = max(0, len(token_ids) - max_prompt_tokens)
    if not dropped:
        return token_ids, 0

    # Keep a small stable prefix for the chat/user header and game identity, then
    # devote the rest of the budget to the newest history and current phase.
    keep_prefix = min(prefix_tokens, max_prompt_tokens // 4)
    keep_recent = max_prompt_tokens - keep_prefix
    retained = token_ids[:keep_prefix] + token_ids[-keep_recent:]
    return retained, dropped


def recent_prompt_token_ids(
    tokenizer, prompt: str, max_prompt_tokens: int, prefix_tokens: int = 256
) -> tuple[List[int], int]:
    """Tokenize a prompt, preserving a stable prefix and its newest history."""
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return _retain_prompt_tokens(token_ids, max_prompt_tokens, prefix_tokens)


def completion_preserving_batch(
    tokenizer,
    observations: Sequence[str],
    actions: Sequence[str],
    max_length: int,
    prefix_tokens: int = 256,
) -> CompletionPreservingBatch:
    """Build a padded causal-LM batch without ever truncating a completion.

    Old observation tokens are removed from the left. The complete sampled
    action remains at the right edge, including its final ``\\boxed{...}``
    payload, so every retained sample has a valid policy-gradient target.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(observations) != len(actions):
        raise ValueError("observations and actions must have equal length")

    observation_ids = tokenizer(
        list(observations), add_special_tokens=False
    )["input_ids"]
    action_ids = tokenizer(list(actions), add_special_tokens=False)["input_ids"]

    sequences, prompt_lengths, sequence_lengths = [], [], []
    original_lengths, prompt_tokens_dropped = [], []
    for obs_ids, act_ids in zip(observation_ids, action_ids):
        if len(act_ids) >= max_length:
            raise ValueError(
                f"completion has {len(act_ids)} tokens but max_length is {max_length}; "
                "increase max_length or reduce generation length"
            )
        retained_obs, dropped = _retain_prompt_tokens(
            obs_ids, max_length - len(act_ids), prefix_tokens
        )
        sequence = retained_obs + act_ids
        sequences.append(sequence)
        prompt_lengths.append(len(retained_obs))
        sequence_lengths.append(len(sequence))
        original_lengths.append(len(obs_ids) + len(act_ids))
        prompt_tokens_dropped.append(dropped)

    encoding = tokenizer.pad(
        {"input_ids": sequences}, padding=True, return_tensors="pt"
    )
    return CompletionPreservingBatch(
        encoding=encoding,
        prompt_lengths=prompt_lengths,
        sequence_lengths=sequence_lengths,
        original_lengths=original_lengths,
        prompt_tokens_dropped=prompt_tokens_dropped,
    )
