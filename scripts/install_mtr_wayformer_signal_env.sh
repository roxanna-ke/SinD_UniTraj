#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sind-mtr-wayformer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
MAMBA_BIN="${MAMBA_BIN:-$HOME/.local/bin/micromamba}"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] env_name=${ENV_NAME}"
echo "[info] python_version=${PYTHON_VERSION}"
echo "[info] torch_index_url=${TORCH_INDEX_URL}"

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[error] missing required command: ${cmd}" >&2
    exit 1
  fi
}

need_cmd bash
need_cmd git
need_cmd curl
need_cmd tar

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[error] nvcc not found. Choose a CUDA devel image or install the CUDA toolkit before running this script." >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "[error] g++ not found. Install build-essential / gcc / g++ first." >&2
  exit 1
fi

mkdir -p "$(dirname "${MAMBA_BIN}")"
if [ ! -x "${MAMBA_BIN}" ]; then
  echo "[step] install micromamba"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o "${tmpdir}/micromamba.tar.bz2"
  tar -xjf "${tmpdir}/micromamba.tar.bz2" -C "${tmpdir}"
  install "${tmpdir}/bin/micromamba" "${MAMBA_BIN}"
fi

eval "$("${MAMBA_BIN}" shell hook -s bash)"

if ! "${MAMBA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[step] create micromamba env"
  "${MAMBA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

micromamba activate "${ENV_NAME}"

echo "[step] upgrade packaging tools"
python -m pip install --upgrade pip setuptools wheel ninja

echo "[step] install pytorch"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo "[step] install runtime dependencies"
python -m pip install -r "${REPO_ROOT}/UniTraj/requirements-mtr-wayformer.txt"

echo "[step] install local scenarionet"
python -m pip install -e "${REPO_ROOT}/scenarionet"

echo "[step] build and install UniTraj"
python -m pip install -e "${REPO_ROOT}/UniTraj"

echo "[step] verify core imports"
python - <<'PY'
import torch
import metadrive
import scenarionet
import unitraj
from unitraj.models.mtr.ops.attention import attention_cuda
from unitraj.models.mtr.ops.knn import knn_cuda

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("metadrive", metadrive.__file__)
print("scenarionet", scenarionet.__file__)
print("unitraj", unitraj.__file__)
print("attention_cuda", attention_cuda.__file__)
print("knn_cuda", knn_cuda.__file__)
PY

cat <<EOF
[done] environment is ready

Activate it with:
  eval "\$(${MAMBA_BIN} shell hook -s bash)"
  micromamba activate ${ENV_NAME}

Recommended next checks:
  python -m sind_converter.scripts.analyze_signal_coverage --help
  python -m sind_converter.scripts.sind_pipeline --help
EOF
