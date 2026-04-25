#!/usr/bin/env bash
#
# HF Jobs runner script. This is the EXACT bash command that
# `hf jobs run --flavor l4x1 ... bash -c <THIS>` executes.
#
# Why this lives outside the Python source tree:
#   - Installs system packages (build-essential for triton's JIT compiler)
#   - Pulls the latest commit of this repo from GitHub at job start time
#   - Patches transformers' quantizer_torchao.py to a no-op stub (avoids the
#     torchao -> register_constant -> torch>=2.7 version creep that took us
#     11+ failed jobs to diagnose)
#   - Sets CC=/usr/bin/gcc so triton can JIT-compile its CUDA driver shim
#
# Submit via the official HF CLI:
#
#   hf jobs run \
#     --flavor l4x1 \
#     --detach \
#     --secrets HF_TOKEN \
#     -e HF_DEBUG=1 \
#     -e PYTHONUNBUFFERED=1 \
#     -e CC=/usr/bin/gcc \
#     --timeout 3h \
#     pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
#     bash -c "$(cat train/hf_job_runner.sh)"
#
# Or with overrides:
#
#   bash train/hf_job_runner.sh   # locally, on a CUDA box

set -ex

echo PHASE_1_apt
apt-get update -qq
apt-get install -y -qq git curl build-essential

echo PHASE_2_clone
# Re-clone in case we're inside HF Jobs (fresh container each run).
# When run locally, /app probably doesn't exist; harmless fallback.
if [ ! -d /app ]; then
    git clone https://github.com/dhruv608/devops-model /app
fi
cd /app

echo PHASE_3_install_uv
pip install --no-cache-dir uv

echo PHASE_3a_install_ml
# All the heavy Python deps. unsloth resolves transformers/trl/peft/accelerate
# transitively when installed last, so order matters: install the explicit
# pins first, then the project, then unsloth.
uv pip install --system --no-cache \
    transformers trl datasets wandb bashlex peft accelerate bitsandbytes

echo PHASE_3b_project
uv pip install --system --no-deps --no-cache -e .

echo PHASE_3c_unsloth
uv pip install --system --no-cache unsloth==2026.4.8

echo PHASE_3d_STUB_torchao
# transformers 5.x hard-imports `quantizer_torchao` from `quantizers/auto.py`.
# Latest torchao requires `torch.utils._pytree.register_constant` (torch 2.7+),
# but our image has torch 2.6. Rather than chase ever-newer torch versions,
# we replace the importer with a no-op class -- transformers loads cleanly,
# we don't use torchao quantization anyway. Surgical and stable.
STUBPATH=/opt/conda/lib/python3.11/site-packages/transformers/quantizers/quantizer_torchao.py
ls -la "$STUBPATH"
echo class TorchAoHfQuantizer:pass > "$STUBPATH"
cat "$STUBPATH"

echo PHASE_3e_uninstall_torchao
# Defense-in-depth: even with the stub, kill the torchao package so any
# transitive `import torchao` elsewhere also fails fast rather than
# silently triggering the register_constant chain.
pip uninstall -y torchao || echo torchao_was_not_installed

echo PHASE_3f_verify_gcc
which gcc
gcc --version

echo PHASE_4_nvidia
nvidia-smi || echo nvidia-smi-missing

echo PHASE_4b_mem
free -h

echo PHASE_5_train
PYTHONPATH=/app \
HF_DEBUG=1 \
PYTHONUNBUFFERED=1 \
CC=/usr/bin/gcc \
python -X faulthandler -u /app/train/train_grpo.py \
    --max_steps 50 \
    --no_vllm \
    --num_generations 2 \
    --max_completion_length 1024 \
    --push_to_hub \
    --hub_model_id dhruv608/safe-sre-grpo-Qwen3-1.7B \
    --report_to none

echo PHASE_5_DONE_RC=$?
