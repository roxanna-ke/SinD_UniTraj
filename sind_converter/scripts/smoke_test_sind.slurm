#!/bin/bash
#SBATCH --job-name=sind_smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-/scratch/izar/ke/sind_raw}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"
SMOKE_ROOT="${SMOKE_ROOT:-${SCRATCH_ROOT}/smoke_test}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${SMOKE_ROOT}/scenarionet_output}"
STRIDE="${STRIDE:-40}"
MAX_RECORDS="${MAX_RECORDS:-2}"
MAX_SCENARIOS_PER_RECORD="${MAX_SCENARIOS_PER_RECORD:-1}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[error] Virtual environment not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "[error] Project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
  echo "[error] Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [ ! -d "${MAP_FALLBACK_ROOT}" ]; then
  echo "[error] Map fallback root not found: ${MAP_FALLBACK_ROOT}" >&2
  exit 1
fi

mkdir -p "${SMOKE_OUTPUT_ROOT}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

LOCAL_CACHE_ROOT="${SCRATCH:-$SCRATCH_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_CACHE_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$LOCAL_CACHE_ROOT/torch}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

echo "[info] host=$(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] smoke_output_root=${SMOKE_OUTPUT_ROOT}"
python --version
which python

srun python - <<'PY'
import importlib

required = ["omegaconf", "numpy", "h5py", "torch", "metadrive"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {', '.join(missing)}")

repo_imports = [
    "sind_converter.data.discovery",
    "sind_converter.scenarios.convert",
    "unitraj.utils.sind_converter",
]
for name in repo_imports:
    importlib.import_module(name)

print("Environment and import smoke check passed.")
PY

echo "[step] minimal conversion smoke test"
srun python UniTraj/unitraj/utils/sind_converter.py \
  --sind-data-root "${DATA_ROOT}" \
  --map-fallback-root "${MAP_FALLBACK_ROOT}" \
  --output-dir "${SMOKE_OUTPUT_ROOT}" \
  --stride "${STRIDE}" \
  --max-records "${MAX_RECORDS}" \
  --max-scenarios-per-record "${MAX_SCENARIOS_PER_RECORD}"

for split in train val; do
  split_dir="${SMOKE_OUTPUT_ROOT}/${split}/sind"
  if [ ! -f "${split_dir}/dataset_summary.pkl" ]; then
    echo "[error] Missing ${split_dir}/dataset_summary.pkl" >&2
    exit 1
  fi
  if [ ! -f "${split_dir}/dataset_mapping.pkl" ]; then
    echo "[error] Missing ${split_dir}/dataset_mapping.pkl" >&2
    exit 1
  fi
done

echo "[done] smoke test passed"
