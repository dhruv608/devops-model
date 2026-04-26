"""Generate the 3 README plot PNGs from plots/eval_results.json.

Hour 21 deliverable. Pure matplotlib, no GPU, no Hub access.

Run: ``PYTHONPATH=. python plots/make_plots.py``
Output: plots/headline_delta.png, plots/per_component_breakdown.png,
        plots/parse_failures.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load() -> dict:
    return json.loads((_THIS_DIR / "eval_results.json").read_text(encoding="utf-8"))


def _save(fig, name: str) -> None:
    out = _THIS_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def headline_delta(d: dict) -> None:
    """Bar chart: 3 headline metrics side by side, untrained vs trained."""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = [
        ("Task success rate", "task_success_rate", "%"),
        ("Safety violation rate", "safety_violation_rate", "%"),
        ("Mean total reward", "mean_total_reward", ""),
    ]
    x = list(range(len(metrics)))
    width = 0.35

    untrained_vals = []
    trained_vals = []
    for _, key, unit in metrics:
        u = d["untrained"][key]
        t = d["trained"][key]
        if unit == "%":
            u *= 100
            t *= 100
        untrained_vals.append(u)
        trained_vals.append(t)

    b1 = ax.bar([xi - width / 2 for xi in x], untrained_vals, width,
                label="Untrained Qwen3-1.7B base", color="#ef4444")
    b2 = ax.bar([xi + width / 2 for xi in x], trained_vals, width,
                label="GRPO-trained (50 steps)", color="#22c55e")

    for b, val, (_, _, unit) in zip(b1, untrained_vals, metrics):
        ax.annotate(f"{val:.2f}{unit}",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for b, val, (_, _, unit) in zip(b2, trained_vals, metrics):
        ax.annotate(f"{val:.2f}{unit}",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_title("Headline metrics — base vs trained on held-out adversarials\n"
                 "(8 scenarios × 3 episodes × 2 models = 48 rollouts)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, "headline_delta.png")


def per_component_breakdown(d: dict) -> None:
    """5 reward components, untrained vs trained."""
    components = [
        ("safety", "mean_safety_reward", -10, 1),
        ("correctness", "mean_correctness_reward", -5, 5),
        ("minimality", "mean_minimality_reward", 0, 1.5),
        ("format", "mean_format_reward", 0, 0.5),
        ("investigation", "mean_investigation_reward", -0.5, 1),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(components)))
    width = 0.35

    u_vals = [d["untrained"][k] for _, k, _, _ in components]
    t_vals = [d["trained"][k] for _, k, _, _ in components]

    ax.bar([xi - width / 2 for xi in x], u_vals, width,
           label="Untrained base", color="#ef4444")
    ax.bar([xi + width / 2 for xi in x], t_vals, width,
           label="GRPO-trained", color="#22c55e")

    # Annotate each with the value
    for xi, u, t in zip(x, u_vals, t_vals):
        ax.annotate(f"{u:+.2f}", (xi - width / 2, u),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
        ax.annotate(f"{t:+.2f}", (xi + width / 2, t),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in components])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Per-component reward — base vs trained\n"
                 "(component caps: safety -10..+1, correctness -5..+5, "
                 "minimality 0..+1.5, format 0..+0.5, investigation -0.5..+1)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, "per_component_breakdown.png")


def parse_failures(d: dict) -> None:
    """The hidden win: 52% drop in parse failures."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Untrained\nQwen3-1.7B base", "GRPO-trained\n(50 steps)"]
    values = [d["untrained"]["parse_failures_total"],
              d["trained"]["parse_failures_total"]]
    n = d["untrained"]["n_episodes"]

    bars = ax.bar(labels, values,
                  color=["#ef4444", "#22c55e"], width=0.5)
    for b, v in zip(bars, values):
        ax.annotate(f"{v} / {n} eps  ({100 * v / n:.0f}%)",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=11)

    ax.set_ylabel("Parse failures (lower is better)")
    delta = values[1] - values[0]
    ax.set_title(f"Parse failures: {values[0]} → {values[1]} "
                 f"({delta:+d}, {100 * delta / values[0]:+.0f}%)\n"
                 "Trained model emits valid JSON tool calls more reliably")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(values) * 1.25)
    _save(fig, "parse_failures.png")


def main() -> int:
    d = _load()
    print(f"loaded eval_results.json: {d['untrained']['n_episodes']} eps × 2 models")
    headline_delta(d)
    per_component_breakdown(d)
    parse_failures(d)
    print("\nall 3 plot PNGs generated under plots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
