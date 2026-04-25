#!/usr/bin/env bash
#
# HF Jobs runner for the post-training eval (Hour 20).
#
# Mirrors train/hf_job_runner.sh's environment setup so we get an
# identical install (image quirks, torchao stub, gcc, openenv from
# GitHub) and then runs `python -m eval.eval` against the 8 held-out
# adversarial scenarios in data/eval_scenarios.json with both the
# untrained Qwen3-1.7B base and our GRPO-trained checkpoint.
#
# Output: plots/eval_results.json (uploaded back to the Hub via the
# same HF Jobs results mechanism, OR we just rely on the trained_model
# repo we already pushed and re-derive after).
#
# Submit:
#
#   hf jobs run \
#     --flavor l4x1 --detach --secrets HF_TOKEN \
#     -e CC=/usr/bin/gcc --timeout 2h \
#     pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
#     bash -c "$(cat eval/run_eval_hf_job.sh)"

set -ex

echo PHASE_E1_apt
apt-get update -qq
apt-get install -y -qq git curl build-essential

echo PHASE_E2_clone
if [ ! -d /app ]; then
    git clone https://github.com/dhruv608/devops-model /app
fi
cd /app

echo PHASE_E3_install_uv
pip install --no-cache-dir uv

echo PHASE_E3a_ml
uv pip install --system --no-cache \
    transformers trl datasets wandb bashlex peft accelerate bitsandbytes

echo PHASE_E3b_openenv
uv pip install --system --no-cache git+https://github.com/meta-pytorch/OpenEnv

echo PHASE_E3c_project
uv pip install --system --no-deps --no-cache -e .

echo PHASE_E3d_unsloth
# unsloth optional for eval (eval doesn't train), but install it so the
# import chain in train_grpo.py / shared modules doesn't blow up.
uv pip install --system --no-cache unsloth==2026.4.8

echo PHASE_E3e_STUB_torchao
STUBPATH=/opt/conda/lib/python3.11/site-packages/transformers/quantizers/quantizer_torchao.py
echo class TorchAoHfQuantizer:pass > "$STUBPATH"
cat "$STUBPATH"

echo PHASE_E3f_uninstall_torchao
pip uninstall -y torchao || echo torchao_was_not_installed

echo PHASE_E4_nvidia
nvidia-smi || echo nvidia-smi-missing

echo PHASE_E5_eval
PYTHONPATH=/app \
HF_DEBUG=1 \
PYTHONUNBUFFERED=1 \
CC=/usr/bin/gcc \
python -X faulthandler -u -m eval.eval \
    --base-model Qwen/Qwen3-1.7B \
    --trained-model dhruv608/safe-sre-grpo-Qwen3-1.7B \
    --episodes-per-scenario 3 \
    --temperature 0.3 \
    --max-new-tokens 256 \
    --out /app/plots/eval_results.json

echo PHASE_E5_DONE_RC=$?

echo PHASE_E6_upload_results
# Echo the JSON so the user can copy from logs even if hub upload fails.
echo "=== eval_results.json ==="
cat /app/plots/eval_results.json | head -200
echo "=== end eval_results.json ==="

# Also push it as a file in the trained model repo so we have a stable URL.
python -X faulthandler -u -c "
import os
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj='/app/plots/eval_results.json',
    path_in_repo='eval_results.json',
    repo_id='dhruv608/safe-sre-grpo-Qwen3-1.7B',
    repo_type='model',
    token=os.environ['HF_TOKEN'],
    commit_message='Add Hour 20 eval_results.json',
)
print('uploaded eval_results.json to dhruv608/safe-sre-grpo-Qwen3-1.7B')
"
