#!/bin/bash
#SBATCH --job-name=sind_scenarionet_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ke/projects/SinD_UniTraj}"
VENV_DIR="${VENV_DIR:-$HOME/projects/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/SinD/Dataset}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${PROJECT_ROOT}/SinD/Data}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"
SCENARIONET_OUTPUT_ROOT="${SCENARIONET_OUTPUT_ROOT:-${SCRATCH_ROOT}/scenarionet_output}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/unitraj_cache}"
STRIDE="${STRIDE:-40}"
CHECK_SAMPLES="${CHECK_SAMPLES:-8}"
DRY_RUN_SAMPLES="${DRY_RUN_SAMPLES:-2}"
MAX_RECORDS="${MAX_RECORDS:-}"
MAX_SCENARIOS_PER_RECORD="${MAX_SCENARIOS_PER_RECORD:-}"
CITIES="${CITIES:-}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found: ${VENV_DIR}" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "Project root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
  echo "SinD dataset root not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [ ! -d "${MAP_FALLBACK_ROOT}" ]; then
  echo "SinD map root not found: ${MAP_FALLBACK_ROOT}" >&2
  exit 1
fi

mkdir -p "${SCENARIONET_OUTPUT_ROOT}" "${CACHE_ROOT}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

LOCAL_CACHE_ROOT="${SCRATCH:-$SCRATCH_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_CACHE_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$LOCAL_CACHE_ROOT/torch}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

python --version
which python

srun python - <<'PY'
import importlib
required = ["omegaconf", "numpy", "h5py", "torch", "metadrive"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {', '.join(missing)}")
print("Environment import check passed.")
PY

CONVERT_ARGS=(
  --sind-data-root "${DATA_ROOT}"
  --map-fallback-root "${MAP_FALLBACK_ROOT}"
  --output-dir "${SCENARIONET_OUTPUT_ROOT}"
  --stride "${STRIDE}"
)

if [ -n "${MAX_RECORDS}" ]; then
  CONVERT_ARGS+=(--max-records "${MAX_RECORDS}")
fi

if [ -n "${MAX_SCENARIOS_PER_RECORD}" ]; then
  CONVERT_ARGS+=(--max-scenarios-per-record "${MAX_SCENARIOS_PER_RECORD}")
fi

if [ -n "${CITIES}" ]; then
  # shellcheck disable=SC2206
  CITY_ARRAY=(${CITIES})
  CONVERT_ARGS+=(--cities "${CITY_ARRAY[@]}")
fi

srun python UniTraj/unitraj/utils/sind_converter.py "${CONVERT_ARGS[@]}"

srun python sind_converter/scripts/build_sind_unitraj_caches.py \
  --train-dir "${SCENARIONET_OUTPUT_ROOT}/train/sind" \
  --val-dir "${SCENARIONET_OUTPUT_ROOT}/val/sind" \
  --cache-root "${CACHE_ROOT}" \
  --methods MTR autobot wayformer \
  --check-samples "${CHECK_SAMPLES}" \
  --dry-run-samples "${DRY_RUN_SAMPLES}" \
  --overwrite
