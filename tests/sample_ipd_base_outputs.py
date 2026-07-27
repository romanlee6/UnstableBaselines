"""Small vLLM smoke test of the revised typed IPD prompts on Qwen3-4B-Base."""

import textarena as ta
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from unstable.utils.templates import (
    apply_template,
    extract_action_and_format_feedback,
)


def phase_prompts():
    env = ta.make("IteratedPrisonersDilemma-Predict-v0-train")
    env.reset(num_players=2, seed=0)

    _, conversation = env.get_observation()
    env.step("[Message: cooperate]")
    env.get_observation()
    env.step("[Message: cooperate]")

    _, prediction = env.get_observation()
    env.step("[Prediction: Cooperate]")
    env.get_observation()
    env.step("[Prediction: Cooperate]")

    _, decision = env.get_observation()
    return {
        "conversation": apply_template("qwen3-multiphase", conversation),
        "prediction": apply_template("qwen3-multiphase", prediction),
        "decision": apply_template("qwen3-multiphase", decision),
    }


def main():
    prompts = phase_prompts()
    labels, batch = [], []
    for phase, prompt in prompts.items():
        for sample_idx in range(4):
            labels.append((phase, sample_idx))
            batch.append(prompt)

    llm = LLM(
        model="Qwen/Qwen3-4B-Base",
        max_model_len=4096,
        max_num_seqs=len(batch),
        disable_custom_all_reduce=True,
    )
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024,
    )
    outputs = llm.generate(batch, params)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Base", local_files_only=True
    )

    print("phase sample tokens finish_reason outer_valid extracted")
    for (phase, sample_idx), output in zip(labels, outputs):
        generated = output.outputs[0]
        raw = generated.text
        extracted, feedback = extract_action_and_format_feedback(raw)
        tokens = len(tokenizer(raw, add_special_tokens=False)["input_ids"])
        print(
            phase,
            sample_idx,
            tokens,
            generated.finish_reason,
            feedback["correct_answer_format"],
            repr(extracted[:120]),
        )


if __name__ == "__main__":
    main()
