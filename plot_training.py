"""Plot training-time reward curves from plots/training_log.jsonl."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "training_log.jsonl")
records = [json.loads(l) for l in open(LOG, encoding="utf-8")]
print(f"Loaded {len(records)} training records.")

if not records:
    raise SystemExit("No training records yet — wait for more steps.")

# Use record index as step (TRL doesn't always print "step", but emits one
# log dict per gradient update, so index is monotonic.)
steps = list(range(1, len(records) + 1))


def col(key, default=None):
    return [r.get(key, default) for r in records]


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Total reward
ax = axes[0, 0]
ax.plot(steps, col("reward", 0), color="#2563eb", linewidth=2)
ax.set_title("Mean training reward (sum of all 5 components)")
ax.set_xlabel("Logged step")
ax.set_ylabel("reward")
ax.grid(True, alpha=0.3)

# 2. Per-component reward
ax = axes[0, 1]
COMPONENTS = ["safety", "correctness", "minimality", "format", "investigation"]
COLORS = ["#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
for comp, color in zip(COMPONENTS, COLORS):
    key = f"rewards/reward_{comp}/mean"
    vals = col(key, 0)
    ax.plot(steps, vals, label=comp, color=color, linewidth=1.8)
ax.set_title("Per-component reward")
ax.set_xlabel("Logged step")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Loss
ax = axes[1, 0]
ax.plot(steps, col("loss", 0), color="#9333ea", linewidth=2)
ax.set_title("GRPO training loss")
ax.set_xlabel("Logged step")
ax.set_ylabel("loss")
ax.grid(True, alpha=0.3)

# 4. Completion length (do completions get shorter / more on-target?)
ax = axes[1, 1]
ax.plot(steps, col("completions/mean_length", 0), color="#65a30d", linewidth=2)
ax.set_title("Mean completion length (tokens)")
ax.set_xlabel("Logged step")
ax.set_ylabel("tokens")
ax.grid(True, alpha=0.3)

plt.suptitle("Safe-SRE GRPO training curves (live, step ≈ {}/150)".format(len(records)),
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = os.path.join(os.path.dirname(LOG), "training_curves.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved -> {out}")

# Also dump a quick text summary
print("\n=== Summary ===")
print(f"Steps logged: {len(records)}")
print(f"First reward: {records[0].get('reward', '?')}")
print(f"Last  reward: {records[-1].get('reward', '?')}")
print(f"Last per-component:")
for comp in COMPONENTS:
    val = records[-1].get(f"rewards/reward_{comp}/mean", "?")
    print(f"  {comp:>14}: {val}")
