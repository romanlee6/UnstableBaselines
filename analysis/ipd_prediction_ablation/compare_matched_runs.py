#!/usr/bin/env python3
"""Compare the matched no-prediction control with the reference Predict run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import wandb


ENTITY = "huaol-massachusetts-institute-of-technology"
PROJECT = "UnstableBaselines"
PREDICT_ENV = "IteratedPrisonersDilemma-Predict-v0-train"
CONTROL_ENV = "IteratedPrisonersDilemma-Broadcast-v0-train"
DEFAULT_PREDICT_RUN = "ub-ipd-typed-recent-200-18994851"
MAX_UPDATE = 100
TAIL = 10


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def history(run_id):
    run = wandb.Api(timeout=90).run(f"{ENTITY}/{PROJECT}/{run_id}")
    return list(run.scan_history(page_size=1000)), {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
    }


def series(rows, key):
    by_step = {}
    for row in rows:
        step, value = row.get("learner/step"), row.get(key)
        if finite(step) and finite(value) and 1 <= int(step) <= MAX_UPDATE:
            by_step[int(step)] = float(value)
    steps = np.array(sorted(by_step), dtype=int)
    return steps, np.array([by_step[step] for step in steps], dtype=float)


def endpoint(values):
    if not len(values):
        return None
    return float(np.mean(values[-min(TAIL, len(values)):]))


def metric_keys(env):
    prefix = f"collection-{env}"
    return {
        "terminal_role_0": f"{prefix}/Reward (pid=0)",
        "terminal_role_1": f"{prefix}/Reward (pid=1)",
        "decision_payoff_role_0": f"{prefix}/role-0/phase/decision/environment_reward",
        "decision_payoff_role_1": f"{prefix}/role-1/phase/decision/environment_reward",
        "mutual_cooperation": f"{prefix}/role-0/phase/decision/mutual_cooperation",
        "conversation_format_role_0": f"{prefix}/role-0/phase/conversation/format/joint",
        "conversation_format_role_1": f"{prefix}/role-1/phase/conversation/format/joint",
        "decision_format_role_0": f"{prefix}/role-0/phase/decision/format/joint",
        "decision_format_role_1": f"{prefix}/role-1/phase/decision/format/joint",
    }


def summarize_run(rows, env):
    summary = {}
    curves = {}
    for label, key in metric_keys(env).items():
        steps, values = series(rows, key)
        curves[label] = (steps, values)
        summary[label] = {
            "last_update": int(steps[-1]) if len(steps) else None,
            "updates_logged": int(len(steps)),
            "endpoint_updates_91_100_or_last_10": endpoint(values),
        }
    return summary, curves


def convergence_signature(curves):
    labels = (
        "decision_payoff_role_0",
        "decision_payoff_role_1",
        "mutual_cooperation",
    )
    maps = {
        label: dict(zip(curves[label][0], curves[label][1]))
        for label in labels
    }
    common = sorted(set.intersection(*(set(values) for values in maps.values())))
    if not common:
        return None
    matrix = np.array([[maps[label][step] for label in labels] for step in common])
    final = np.mean(matrix[-min(TAIL, len(matrix)):], axis=0)
    distance = np.linalg.norm(matrix - final, axis=1)
    scale = max(float(distance[0]), 1e-9)
    normalized = distance / scale
    for index in range(max(0, len(common) - 4)):
        if np.all(normalized[index:index + 5] <= 0.15):
            update = int(common[index])
            return {
                "update": update,
                "shared_phase_samples_per_role": update * 128,
            }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", required=True, help="W&B run ID for the matched control")
    parser.add_argument("--predict-run", default=DEFAULT_PREDICT_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "matched_comparison.json",
    )
    args = parser.parse_args()

    predict_rows, predict_meta = history(args.predict_run)
    control_rows, control_meta = history(args.control_run)
    predict_summary, predict_curves = summarize_run(predict_rows, PREDICT_ENV)
    control_summary, control_curves = summarize_run(control_rows, CONTROL_ENV)
    result = {
        "comparison_horizon": "learner updates 1-100",
        "endpoint": "mean of updates 91-100, or last 10 available values",
        "convergence": (
            "first five-update window within 15% of the final "
            "[role-0 payoff, role-1 payoff, mutual-cooperation] signature"
        ),
        "matched_training_contract": {
            "shared_phase_samples_per_role_per_update": 128,
            "shared_phase_loss_weight": 1.0 / 3.0,
            "control_batch_size": 256,
            "predict_batch_size": 384,
        },
        "predict": {
            "run": predict_meta,
            "metrics": predict_summary,
            "convergence": convergence_signature(predict_curves),
        },
        "control": {
            "run": control_meta,
            "metrics": control_summary,
            "convergence": convergence_signature(control_curves),
        },
        "limitations": [
            "This is a single-run descriptive comparison without seed-level uncertainty.",
            "The control omits prediction calls, so wall-clock time and total generated completions are intentionally lower.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
