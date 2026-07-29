#!/usr/bin/env python3
"""Download three W&B histories and render the IPD run comparison.

The legacy vanilla and broadcast runs did not log actions. Their equilibrium
classification is therefore limited to terminal outcome symmetry/asymmetry.
The prediction run additionally logs mutual cooperation and prediction
accuracy, allowing its tied outcome to be identified as mutual cooperation.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    # Dependencies were installed here temporarily for the original analysis.
    sys.path.insert(0, "/tmp/ipd_plotdeps")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

import wandb


ENTITY = "huaol-massachusetts-institute-of-technology"
PROJECT = "UnstableBaselines"
RUNS = {
    "Vanilla": {
        "id": "xjnxzwn9",
        "env": "IteratedPrisonersDilemma-v0-train",
        "opponent": "Gemini 3.1 Flash Lite",
        "color": "#4C78A8",
    },
    "Broadcast": {
        "id": "asl7pl50",
        "env": "IteratedPrisonersDilemma-Broadcast-v0-train",
        "opponent": "Gemini 3.1 Flash Lite",
        "color": "#F58518",
    },
    "Predict": {
        "id": "ub-ipd-typed-recent-200-18994851",
        "env": "IteratedPrisonersDilemma-Predict-v0-train",
        "opponent": "DeepSeek-V4-flash",
        "color": "#54A24B",
    },
}

OUT = Path(__file__).resolve().parent
CACHE = OUT / "wandb_histories.json.gz"


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def download_histories():
    api = wandb.Api(timeout=90)
    histories = {}
    metadata = {}
    for label, spec in RUNS.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{spec['id']}")
        histories[label] = list(run.scan_history(page_size=1000))
        metadata[label] = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
        }
    with gzip.open(CACHE, "wt", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "histories": histories}, handle)
    return histories, metadata


def load_data(refresh=False):
    if refresh or not CACHE.exists():
        return download_histories()
    with gzip.open(CACHE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["histories"], payload["metadata"]


def series(rows, key):
    """Return the last logged metric value at each learner update."""
    by_step = {}
    for row in rows:
        step, value = row.get("learner/step"), row.get(key)
        if finite(step) and finite(value):
            by_step[int(step)] = float(value)
    steps = np.array(sorted(by_step), dtype=float)
    values = np.array([by_step[int(step)] for step in steps], dtype=float)
    return steps, values


def rolling(values, window=5):
    if len(values) == 0:
        return values
    out = np.empty_like(values)
    for index in range(len(values)):
        out[index] = np.mean(values[max(0, index - window + 1) : index + 1])
    return out


def aligned_pair(rows, key0, key1):
    x0, y0 = series(rows, key0)
    x1, y1 = series(rows, key1)
    d0 = dict(zip(x0.astype(int), y0))
    d1 = dict(zip(x1.astype(int), y1))
    common = np.array(sorted(set(d0) & set(d1)), dtype=float)
    return common, np.array([d0[int(x)] for x in common]), np.array([d1[int(x)] for x in common])


def endpoint(values, tail=10):
    return float(np.mean(values[-min(tail, len(values)) :]))


def convergence_step(steps, distances, threshold=0.15, hold=5):
    """First update beginning a stable run below 15% endpoint distance."""
    if not len(distances):
        return None
    scale = max(float(distances[0]), 1e-9)
    normalized = distances / scale
    for i in range(max(0, len(normalized) - hold + 1)):
        if np.all(normalized[i : i + hold] <= threshold):
            return int(steps[i])
    return None


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_keys(label):
    env = RUNS[label]["env"]
    prefix = f"collection-{env}"
    return f"{prefix}/Reward (pid=0)", f"{prefix}/Reward (pid=1)"


def figure_learning(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    convergence = {}
    endpoints = {}
    for label, spec in RUNS.items():
        key0, key1 = metric_keys(label)
        steps, role0, role1 = aligned_pair(histories[label], key0, key1)
        role0_s, role1_s = rolling(role0), rolling(role1)
        axes[0].plot(steps, role0_s, color=spec["color"], lw=2, label=f"{label}: role 0")
        axes[0].plot(steps, role1_s, color=spec["color"], lw=1.6, ls="--", label=f"{label}: role 1")

        final = np.array([endpoint(role0), endpoint(role1)])
        distances = np.sqrt((role0_s - final[0]) ** 2 + (role1_s - final[1]) ** 2)
        denom = max(float(distances[0]), 1e-9)
        normalized = distances / denom
        conv = convergence_step(steps, distances)
        convergence[label] = conv
        endpoints[label] = final.tolist()
        axes[1].plot(steps, np.maximum(normalized, 0.008), color=spec["color"], lw=2, label=label)
        if conv is not None:
            axes[1].scatter([conv], [normalized[list(steps.astype(int)).index(conv)]], color=spec["color"], s=35, zorder=5)
            axes[1].annotate(f"{conv}", (conv, 0.16), color=spec["color"], ha="center", va="bottom")

    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set(title="A. Self-play terminal outcomes", xlabel="Learner update", ylabel="Mean terminal reward (win=1, draw=0, loss=−1)", ylim=(-1.08, 1.08))
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(0.15, color="#777777", lw=1, ls=":", label="15% threshold")
    axes[1].set(
        title="B. Distance to each run’s final outcome signature",
        xlabel="Learner update",
        ylabel="Normalized distance (initial = 1; log scale)",
        ylim=(0.008, 8),
    )
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.suptitle("Learning speed toward the observed self-play attractor", y=1.02, fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Curves are 5-update moving means. Convergence is the first 5-update span within 15% of the final 10-update outcome vector.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "figure_1_learning_speed")
    return convergence, endpoints


def figure_equilibrium(histories):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for label, spec in RUNS.items():
        key0, key1 = metric_keys(label)
        steps, role0, role1 = aligned_pair(histories[label], key0, key1)
        role0, role1 = rolling(role0), rolling(role1)
        stride = max(1, len(steps) // 30)
        ax.plot(role0, role1, color=spec["color"], alpha=0.65, lw=1.5)
        ax.scatter(role0[::stride], role1[::stride], c=steps[::stride], cmap="Greys", s=12, alpha=0.35)
        x, y = endpoint(role0), endpoint(role1)
        ax.scatter([x], [y], color=spec["color"], s=90, edgecolor="white", linewidth=1.2, zorder=5)
        suffix = ""
        if label == "Predict":
            prefix = f"collection-{spec['env']}"
            _, cc = series(histories[label], f"{prefix}/role-0/phase/decision/mutual_cooperation")
            suffix = f"\nCC={endpoint(cc):.1%} (directly logged)"
        elif label == "Vanilla":
            suffix = "\ntie; CC vs DD not logged"
        else:
            suffix = "\nrole 1 dominates"
        offsets = {
            "Broadcast": (18, -5),
            "Vanilla": (18, -4),
            "Predict": (18, 26),
        }
        ax.annotate(
            f"{label}{suffix}",
            (x, y),
            xytext=offsets[label],
            textcoords="offset points",
            color=spec["color"],
            fontsize=9,
        )

    ax.plot([-1, 1], [1, -1], color="#999999", lw=1, ls=":", label="zero-sum terminal outcomes")
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.set(
        title="Observed self-play outcome attractors",
        xlabel="Role 0 terminal reward",
        ylabel="Role 1 terminal reward",
        xlim=(-1.08, 1.08),
        ylim=(-1.08, 1.08),
        aspect="equal",
    )
    ax.text(0.68, -0.92, "role 0 dominates", color="#666666", fontsize=8, ha="center")
    ax.text(0.03, 0.03, "symmetric/tied", color="#666666", fontsize=8)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "figure_2_equilibrium_type")


def eval_key(label):
    env = RUNS[label]["env"]
    if label in ("Vanilla", "Broadcast"):
        opponent = "google/gemini-3.1-flash-lite-preview-20260303"
        return f"evaluation-{env} ({opponent})/Reward"
    return f"evaluation-{env}/Reward"


def figure_unseen(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), gridspec_kw={"width_ratios": [2.1, 1]})
    final_eval = {}
    for label, spec in RUNS.items():
        steps, reward = series(histories[label], eval_key(label))
        reward_s = rolling(reward)
        axes[0].plot(steps, reward_s, lw=2.2, color=spec["color"], label=f"{label} vs {spec['opponent']}")
        final_eval[label] = endpoint(reward)
    axes[0].axhline(0, color="#777777", lw=0.9)
    axes[0].set(
        title="A. Role 0 against a fixed unseen opponent",
        xlabel="Learner update",
        ylabel="Mean terminal reward",
        ylim=(-1.08, 1.08),
    )
    axes[0].legend(fontsize=8)

    labels = list(RUNS)
    values = [final_eval[label] for label in labels]
    colors = [RUNS[label]["color"] for label in labels]
    bars = axes[1].bar(labels, values, color=colors, width=0.68)
    axes[1].axhline(0, color="#777777", lw=0.9)
    axes[1].bar_label(bars, labels=[f"{value:+.2f}" for value in values], padding=3, fontsize=9)
    axes[1].set(title="B. Final 10-update mean", ylabel="Mean terminal reward", ylim=(-1.12, 0.25))
    fig.suptitle("Generalization to unseen agents", y=1.02, fontsize=13)
    fig.text(
        0.5,
        -0.025,
        "Opponent is not controlled across runs: Vanilla/Broadcast use Gemini; Predict uses DeepSeek-V4-flash. Compare trends cautiously.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "figure_3_unseen_agent")
    return final_eval


def figure_predict_mechanism(histories):
    label = "Predict"
    spec = RUNS[label]
    rows = histories[label]
    prefix = f"collection-{spec['env']}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    phase_colors = {"conversation": "#9C755F", "prediction": "#B279A2", "decision": "#59A14F"}
    for phase in ("conversation", "prediction", "decision"):
        role_curves = []
        common_steps = None
        for role in (0, 1):
            steps, values = series(rows, f"{prefix}/role-{role}/phase/{phase}/format/joint")
            if common_steps is None:
                common_steps = steps
            role_curves.append(values)
        length = min(map(len, role_curves))
        mean_curve = np.mean(np.vstack([curve[:length] for curve in role_curves]), axis=0)
        axes[0].plot(common_steps[:length], rolling(mean_curve), lw=2, color=phase_colors[phase], label=phase.capitalize())
    axes[0].set(title="A. Typed output validity", xlabel="Learner update", ylabel="Joint format-valid rate", ylim=(0, 1.04))
    axes[0].legend(fontsize=8)

    for role, ls in ((0, "-"), (1, "--")):
        steps, accuracy = series(rows, f"{prefix}/role-{role}/phase/prediction/accuracy")
        axes[1].plot(steps, rolling(accuracy), color="#B279A2", ls=ls, lw=2, label=f"Prediction accuracy, role {role}")
    steps, cc = series(rows, f"{prefix}/role-0/phase/decision/mutual_cooperation")
    axes[1].plot(steps, rolling(cc), color="#59A14F", lw=2.4, label="Mutual cooperation")
    axes[1].set(title="B. ToM prediction and coordinated action", xlabel="Learner update", ylabel="Rate", ylim=(0, 1.04))
    axes[1].legend(fontsize=8)

    for phase in ("prediction", "decision"):
        curves = []
        phase_steps = None
        for role in (0, 1):
            phase_steps, values = series(rows, f"{prefix}/role-{role}/phase/{phase}/environment_reward")
            curves.append(values)
        length = min(map(len, curves))
        axes[2].plot(
            phase_steps[:length],
            rolling(np.mean(np.vstack([curve[:length] for curve in curves]), axis=0)),
            color=phase_colors[phase],
            lw=2,
            label=f"{phase.capitalize()} reward",
        )
    axes[2].axhline(3, color="#888888", ls=":", lw=1, label="CC payoff = 3")
    axes[2].set(title="C. Phase-specific environment reward", xlabel="Learner update", ylabel="Mean reward at phase", ylim=(-0.05, 3.35))
    axes[2].legend(fontsize=8)

    fig.suptitle("Why the three-phase prediction environment learns a distinct pattern", y=1.02, fontsize=13)
    fig.text(
        0.5,
        -0.025,
        "Conversation learns only typed-format reward; prediction receives accuracy bonus (0/1); decision receives IPD payoff (0/1/3/5).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "figure_4_predict_phases")


def summarize(histories, metadata, convergence, endpoints, final_eval):
    predict_prefix = f"collection-{RUNS['Predict']['env']}"
    _, cc = series(histories["Predict"], f"{predict_prefix}/role-0/phase/decision/mutual_cooperation")
    _, acc0 = series(histories["Predict"], f"{predict_prefix}/role-0/phase/prediction/accuracy")
    _, acc1 = series(histories["Predict"], f"{predict_prefix}/role-1/phase/prediction/accuracy")

    summary = {
        "convergence_update_15pct": convergence,
        "self_play_terminal_reward_endpoint": endpoints,
        "unseen_agent_reward_endpoint": final_eval,
        "predict_mutual_cooperation_endpoint": endpoint(cc),
        "predict_prediction_accuracy_endpoint": {
            "role_0": endpoint(acc0),
            "role_1": endpoint(acc1),
        },
        "run_metadata": metadata,
        "limitations": [
            "Vanilla and Broadcast do not log actions, so symmetric terminal outcomes cannot distinguish CC from DD.",
            "Unseen opponents differ: Vanilla/Broadcast use Gemini 3.1 Flash Lite; Predict uses DeepSeek-V4-flash.",
            "All curves are single-run descriptive results; no seed-level uncertainty intervals are available.",
            "Logged collection metrics are rolling means over up to 512 samples, so transitions are intentionally lagged.",
            "The Predict cooperate/defect scalar keys are malformed by a tracker substring check and are excluded; mutual_cooperation comes from environment step_info.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Three-run IPD comparison",
        "",
        "## Main findings",
        "",
        f"- **Learning speed:** first stable entry within 15% of the final outcome signature: "
        + ", ".join(f"{label} update {step}" if step is not None else f"{label} not reached" for label, step in convergence.items())
        + ".",
        f"- **Self-play equilibrium signature:** Vanilla ends near ({endpoints['Vanilla'][0]:+.2f}, {endpoints['Vanilla'][1]:+.2f}) "
        f"and Predict near ({endpoints['Predict'][0]:+.2f}, {endpoints['Predict'][1]:+.2f}), both symmetric/tied. "
        f"Predict is directly identified as mutual cooperation ({endpoint(cc):.1%} CC). Vanilla cannot be split into CC versus DD from its logs.",
        f"- **Broadcast equilibrium:** ends near ({endpoints['Broadcast'][0]:+.2f}, {endpoints['Broadcast'][1]:+.2f}); "
        "role 1 almost always wins. Removing within-turn message ordering therefore did not remove the learned role asymmetry in this run.",
        f"- **Unseen-agent evaluation:** final terminal reward is Vanilla {final_eval['Vanilla']:+.2f}, "
        f"Broadcast {final_eval['Broadcast']:+.2f}, Predict {final_eval['Predict']:+.2f}. "
        "Predict is close to neutral while both baselines lose, but its opponent differs, so this is suggestive rather than a controlled ranking.",
        f"- **Prediction mechanism:** final prediction accuracy is {endpoint(acc0):.1%} for role 0 and {endpoint(acc1):.1%} for role 1, "
        f"alongside {endpoint(cc):.1%} mutual cooperation. Phase-specific rewards make typed communication, opponent modeling, and action choice separate learning problems.",
        "",
        "## Interpretation caveats",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "1. `figure_1_learning_speed`: outcome curves and normalized distance to final attractor.",
            "2. `figure_2_equilibrium_type`: role-pair terminal outcome trajectories and endpoints.",
            "3. `figure_3_unseen_agent`: fixed-unseen-opponent evaluation curves and endpoint bars.",
            "4. `figure_4_predict_phases`: phase validity, prediction/coordination, and phase reward channels.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    refresh = "--refresh" in sys.argv
    histories, metadata = load_data(refresh=refresh)
    style()
    convergence, endpoints = figure_learning(histories)
    figure_equilibrium(histories)
    final_eval = figure_unseen(histories)
    figure_predict_mechanism(histories)
    summarize(histories, metadata, convergence, endpoints, final_eval)
    print(f"Wrote comparison to {OUT}")


if __name__ == "__main__":
    main()
