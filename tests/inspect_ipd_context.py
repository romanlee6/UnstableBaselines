"""Measure IPD prompt growth and verify recent-history retention."""

import textarena as ta
from transformers import AutoTokenizer

from unstable.utils.context_window import (
    completion_preserving_batch,
    recent_prompt_token_ids,
)
from unstable.utils.templates import apply_template


MAX_TRAIN_TOKENS = 4096
MAX_GENERATION_TOKENS = 1024
MAX_PROMPT_TOKENS = MAX_TRAIN_TOKENS - MAX_GENERATION_TOKENS


def collect_decision_prompts(message_words: int):
    env = ta.make("IteratedPrisonersDilemma-Predict-v0-train")
    env.reset(num_players=2, seed=0)
    prompts = []
    filler = " ".join(["context"] * message_words)

    for round_number in range(1, 11):
        for action in (
            f"[Message: role 0 round {round_number} {filler}]",
            f"[Message: role 1 round {round_number} {filler}]",
            "[Prediction: Cooperate]",
            "[Prediction: Cooperate]",
            "[Action: Cooperate]",
        ):
            env.get_observation()
            done, _ = env.step(action)
            assert not done

        pid, observation = env.get_observation()
        assert pid == 1
        assert "Round 1 results:" in observation or round_number == 1
        assert f"CURRENT PHASE: DECISION (round {round_number})" in observation
        prompts.append(apply_template("qwen3-multiphase", observation))

        done, _ = env.step("[Action: Cooperate]")
        assert done == (round_number == 10)
    return prompts


def inspect_scenario(tokenizer, name: str, message_words: int):
    prompts = collect_decision_prompts(message_words)
    completion = r"\boxed{[Action: Cooperate]}"
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    print(f"scenario={name} message_words={message_words}")
    print("round original_prompt_tokens collection_tokens dropped train_tokens")

    for round_number, prompt in enumerate(prompts, start=1):
        original_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        collection_ids, collection_dropped = recent_prompt_token_ids(
            tokenizer, prompt, MAX_PROMPT_TOKENS
        )
        assert len(collection_ids) <= MAX_PROMPT_TOKENS
        if collection_dropped:
            # The newest phase instruction and assistant-generation prefix live
            # in the retained suffix.
            retained_text = tokenizer.decode(
                collection_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            assert f"CURRENT PHASE: DECISION (round {round_number})" in retained_text

        effective_prompt = (
            prompt
            if not collection_dropped
            else tokenizer.decode(
                collection_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

        prepared = completion_preserving_batch(
            tokenizer, [effective_prompt], [completion], MAX_TRAIN_TOKENS
        )
        retained_sequence = prepared.encoding.input_ids[0][
            :prepared.sequence_lengths[0]
        ].tolist()
        assert retained_sequence[-len(completion_ids):] == completion_ids
        assert prepared.sequence_lengths[0] <= MAX_TRAIN_TOKENS

        print(
            round_number,
            len(original_ids),
            len(collection_ids),
            collection_dropped,
            prepared.sequence_lengths[0],
        )


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Base", local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"tokenizer.truncation_side={tokenizer.truncation_side}")
    print(
        f"settings train={MAX_TRAIN_TOKENS} generation={MAX_GENERATION_TOKENS} "
        f"prompt={MAX_PROMPT_TOKENS}"
    )
    inspect_scenario(tokenizer, "concise", message_words=8)
    inspect_scenario(tokenizer, "long_messages", message_words=256)


if __name__ == "__main__":
    main()
