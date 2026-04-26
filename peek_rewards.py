"""Pull live training logs, parse the GRPOTrainer per-step JSON, write them
to plots/training_log.jsonl (one record per step) so we can plot later."""
import json
import os
import re
import sys

from huggingface_hub import HfApi

# Force ASCII-safe stdout on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOB_ID = "69ed8335d70108f37acdf673"
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

print(f"Fetching logs for job {JOB_ID}...")
logs = list(api.fetch_job_logs(job_id=JOB_ID, namespace="dhruv608"))
print(f"Got {len(logs)} log lines total.")

# GRPOTrainer prints dicts like:
#   {'loss': 0.123, 'grad_norm': 0.5, ..., 'reward': 1.92, 'epoch': 0.1}
# These are our training-curve datapoints.
DICT_RE = re.compile(r"\{[^\{\}]*'(?:loss|reward|epoch)'[^\{\}]*\}")

records = []
for raw in logs:
    s = str(raw)
    for m in DICT_RE.finditer(s):
        try:
            # Logged with single quotes, so swap to JSON-safe
            parsed = json.loads(m.group(0).replace("'", '"'))
            if isinstance(parsed, dict) and ("loss" in parsed or "reward" in parsed):
                records.append(parsed)
        except json.JSONDecodeError:
            pass

print(f"Parsed {len(records)} per-step training records.\n")
if records:
    last = records[-1]
    print("Most recent step summary:")
    for k, v in sorted(last.items()):
        print(f"  {k}: {v}")

# Persist for later plotting
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "training_log.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"\nWrote {len(records)} records -> {out_path}")
