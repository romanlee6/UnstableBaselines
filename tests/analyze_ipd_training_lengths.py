"""Token-length diagnostics for the previous IPD phase-balanced run."""

import csv
import glob
import ast
from collections import defaultdict

from transformers import AutoTokenizer


RUN_GLOB = (
    "outputs/*/*/"
    "UB-multirole-reinforce-Qwen3-4B-Base-"
    "IteratedPrisonersDilemma-Predict-v0-train-phase-balanced-azure-18228672/"
    "training_data/train_data_step_*_role_*.csv"
)
MAX_SAMPLES_PER_PHASE = 12_000


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def tokenize_lengths(tokenizer, texts, batch_size=512):
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], add_special_tokens=False
        )["input_ids"]
        lengths.extend(map(len, encoded))
    return lengths


def summarize(name, values):
    return (
        f"{name}: n={len(values)} p50={percentile(values, 0.50)} "
        f"p90={percentile(values, 0.90)} p95={percentile(values, 0.95)} "
        f"p99={percentile(values, 0.99)} max={max(values)}"
    )


def main():
    paths = sorted(glob.glob(RUN_GLOB))
    if not paths:
        raise RuntimeError(f"No training CSV files matched {RUN_GLOB}")

    samples = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                phase = row["phase"]
                if len(samples[phase]) < MAX_SAMPLES_PER_PHASE:
                    components = ast.literal_eval(row["reward_components"])
                    samples[phase].append((
                        row["obs"],
                        row["act"],
                        bool(components.get("outer_format_valid")),
                        bool(components.get("joint_format_valid")),
                    ))
        if all(len(samples[phase]) >= MAX_SAMPLES_PER_PHASE for phase in (
            "conversation", "prediction", "decision"
        )):
            break

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Base", local_files_only=True
    )
    print(f"files_scanned={len(paths)} sample_cap_per_phase={MAX_SAMPLES_PER_PHASE}")
    for phase in ("conversation", "prediction", "decision"):
        phase_samples = samples[phase]
        observations = [obs for obs, _, _, _ in phase_samples]
        actions = [act for _, act, _, _ in phase_samples]
        outer_valid = [valid for _, _, valid, _ in phase_samples]
        joint_valid = [valid for _, _, _, valid in phase_samples]
        obs_lengths = tokenize_lengths(tokenizer, observations)
        action_lengths = tokenize_lengths(tokenizer, actions)
        combined_lengths = [
            obs_len + action_len
            for obs_len, action_len in zip(obs_lengths, action_lengths)
        ]
        print(f"phase={phase}")
        print(summarize("  observation", obs_lengths))
        print(summarize("  raw_action", action_lengths))
        print(summarize("  combined", combined_lengths))
        print(
            f"  action_ge_1024={sum(length >= 1024 for length in action_lengths)} "
            f"combined_gt_3000={sum(length > 3000 for length in combined_lengths)} "
            f"combined_gt_4096={sum(length > 4096 for length in combined_lengths)}"
        )
        ceiling = [length >= 1024 for length in action_lengths]
        print(
            f"  outer_valid={sum(outer_valid)} joint_valid={sum(joint_valid)} "
            f"outer_valid_at_ceiling={sum(valid and hit for valid, hit in zip(outer_valid, ceiling))} "
            f"ceiling_samples={sum(ceiling)}"
        )
        valid_lengths = [
            length for length, valid in zip(action_lengths, outer_valid) if valid
        ]
        invalid_lengths = [
            length for length, valid in zip(action_lengths, outer_valid) if not valid
        ]
        if valid_lengths:
            print(summarize("  outer_valid_action", valid_lengths))
        if invalid_lengths:
            print(summarize("  outer_invalid_action", invalid_lengths))


if __name__ == "__main__":
    main()
