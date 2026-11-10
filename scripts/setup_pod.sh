#!/usr/bin/env bash
# Prepares a fresh CUDA pod for the sweep: uv, Goodfire's steering fork of vLLM, the sweep's dependencies, the
# linear-attention kernels HF uses (fla-core), and the model weights. Everything installs under /workspace, so a
# stopped pod restarts ready. The fork's additions are Python + Triton, so it installs on top of the upstream wheel
# built at the commit the fork last merged (755492e3).
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cat > /workspace/env.sh <<'EOF'
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/.uv-cache UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
export UV_INSTALL_DIR=/workspace/bin PATH=/workspace/bin:$HOME/.local/bin:$PATH
EOF
grep -q /workspace/env.sh ~/.bashrc || echo "source /workspace/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh

command -v rsync >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uvx --from huggingface_hub hf download "$MODEL" > /workspace/download.log 2>&1 &
DL=$!

[ -d vllm-steer ] || git clone --filter=blob:none https://github.com/RhizoNymph/vllm vllm-steer
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
(cd vllm-steer && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=755492e37d7d7201d93f0effa200acac56601f38 \
    uv pip install -e . --torch-backend=auto)
uv pip install datasets accelerate safetensors numpy
uv pip install --no-deps flash-linear-attention fla-core einops
wait $DL
uv run python -c "import vllm, transformers, torch, fla.ops.gated_delta_rule; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| fla ok | cuda', torch.cuda.get_device_name())"
