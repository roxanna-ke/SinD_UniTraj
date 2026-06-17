#!/bin/bash
#SBATCH --job-name=sind_scenarionet_cache
#SBATCH --output=/home/%u/projects/SinD_UniTraj/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-/scratch/izar/ke/sind_raw}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${SCRATCH_ROOT}/canonical_scenarios}"
SPLIT_ROOT="${SPLIT_ROOT:-${SCRATCH_ROOT}/splits}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/unitraj_cache}"
SPLIT_MODE="${SPLIT_MODE:-record-level}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SPLIT_SEED="${SPLIT_SEED:-42}"
HELDOUT_CITIES="${HELDOUT_CITIES:-}"
TRAIN_CITIES="${TRAIN_CITIES:-}"
TEST_CITIES="${TEST_CITIES:-}"
STRIDE="${STRIDE:-40}"
CHECK_SAMPLES="${CHECK_SAMPLES:-8}"
DRY_RUN_SAMPLES="${DRY_RUN_SAMPLES:-2}"
MAX_RECORDS="${MAX_RECORDS:-}"
MAX_SCENARIOS_PER_RECORD="${MAX_SCENARIOS_PER_RECORD:-}"
CITIES="${CITIES:-}"
TRAINING_MAPPING_TABLE="${TRAINING_MAPPING_TABLE:-${PROJECT_ROOT}/osm_audit_four_cities/osm_training_mapping_table.csv}"

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

if [ ! -f "${TRAINING_MAPPING_TABLE}" ]; then
  echo "OSM training mapping table not found: ${TRAINING_MAPPING_TABLE}" >&2
  echo "Run sind_pipeline.py audit-maps first, or set TRAINING_MAPPING_TABLE to the audited CSV." >&2
  exit 1
fi

mkdir -p "${CANONICAL_ROOT}" "${SPLIT_ROOT}" "${CACHE_ROOT}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

LOCAL_CACHE_ROOT="${SCRATCH:-$SCRATCH_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_CACHE_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$LOCAL_CACHE_ROOT/torch}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

echo "[info] host=$(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] map_fallback_root=${MAP_FALLBACK_ROOT}"
echo "[info] training_mapping_table=${TRAINING_MAPPING_TABLE}"
echo "[info] canonical_root=${CANONICAL_ROOT}"
echo "[info] split_root=${SPLIT_ROOT}"
echo "[info] cache_root=${CACHE_ROOT}"
echo "[info] split_mode=${SPLIT_MODE}"
echo "[info] stride=${STRIDE}"
python --version
which python

srun python - <<'PY'
import importlib
required = ["omegaconf", "numpy", "h5py", "torch", "metadrive"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {', '.join(missing)}")
for name in ["sind_converter.scenarios.convert", "sind_converter.splits.make"]:
    importlib.import_module(name)
print("Environment import check passed.")
PY

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --map-fallback-root "${MAP_FALLBACK_ROOT}"
  --canonical-scenario-root "${CANONICAL_ROOT}"
  --split-root "${SPLIT_ROOT}"
  --cache-root "${CACHE_ROOT}"
  --training-mapping-table "${TRAINING_MAPPING_TABLE}"
  --stride "${STRIDE}"
)
CONVERT_EXTRA_ARGS=()

if [ -n "${MAX_RECORDS}" ]; then
  CONVERT_EXTRA_ARGS+=(--max-records "${MAX_RECORDS}")
fi

if [ -n "${MAX_SCENARIOS_PER_RECORD}" ]; then
  CONVERT_EXTRA_ARGS+=(--max-scenarios-per-record "${MAX_SCENARIOS_PER_RECORD}")
fi

if [ -n "${CITIES}" ]; then
  # shellcheck disable=SC2206
  CITY_ARRAY=(${CITIES})
  CONVERT_EXTRA_ARGS+=(--cities "${CITY_ARRAY[@]}")
fi

echo "[step] canonical ScenarioNet conversion"
srun python -m sind_converter.scripts.sind_pipeline convert-scenarios \
  "${COMMON_ARGS[@]}" \
  "${CONVERT_EXTRA_ARGS[@]}"

CANONICAL_DIR="${CANONICAL_ROOT}/sind"
for file_name in dataset_summary.pkl dataset_mapping.pkl; do
  if [ ! -f "${CANONICAL_DIR}/${file_name}" ]; then
    echo "[error] Missing ${CANONICAL_DIR}/${file_name}" >&2
    exit 1
  fi
done

SPLIT_ARGS=(
  --canonical-root "${CANONICAL_DIR}"
  --mode "${SPLIT_MODE}"
)

if [ "${SPLIT_MODE}" = "record-level" ]; then
  SPLIT_ARGS+=(--train-ratio "${TRAIN_RATIO}" --seed "${SPLIT_SEED}")
  TRAIN_SPLIT_DIR="${SPLIT_ROOT}/record_level/train/sind"
  VAL_SPLIT_DIR="${SPLIT_ROOT}/record_level/test/sind"
else
  if [ -n "${HELDOUT_CITIES}" ]; then
    # shellcheck disable=SC2206
    HELDOUT_CITY_ARRAY=(${HELDOUT_CITIES})
    SPLIT_ARGS+=(--heldout-cities "${HELDOUT_CITY_ARRAY[@]}")
  fi
  if [ -n "${TRAIN_CITIES}" ]; then
    # shellcheck disable=SC2206
    TRAIN_CITY_ARRAY=(${TRAIN_CITIES})
    SPLIT_ARGS+=(--train-cities "${TRAIN_CITY_ARRAY[@]}")
  fi
  if [ -n "${TEST_CITIES}" ]; then
    # shellcheck disable=SC2206
    TEST_CITY_ARRAY=(${TEST_CITIES})
    SPLIT_ARGS+=(--test-cities "${TEST_CITY_ARRAY[@]}")
  fi
  TRAIN_SPLIT_DIR="${SPLIT_ROOT}/city_holdout/train/sind"
  VAL_SPLIT_DIR="${SPLIT_ROOT}/city_holdout/test/sind"
fi

echo "[step] ScenarioNet split generation"
srun python -m sind_converter.scripts.sind_pipeline make-splits \
  "${COMMON_ARGS[@]}" \
  "${SPLIT_ARGS[@]}"

for split_dir in "${TRAIN_SPLIT_DIR}" "${VAL_SPLIT_DIR}"; do
  for file_name in dataset_summary.pkl dataset_mapping.pkl; do
    if [ ! -f "${split_dir}/${file_name}" ]; then
      echo "[error] Missing ${split_dir}/${file_name}" >&2
      exit 1
    fi
  done
done

echo "[step] UniTraj cache validation and build"
srun python -m sind_converter.scripts.build_sind_unitraj_caches \
  --train-dir "${TRAIN_SPLIT_DIR}" \
  --val-dir "${VAL_SPLIT_DIR}" \
  --cache-root "${CACHE_ROOT}" \
  --methods MTR autobot wayformer \
  --check-samples "${CHECK_SAMPLES}" \
  --dry-run-samples "${DRY_RUN_SAMPLES}" \
  --overwrite

for method in MTR autobot wayformer; do
  for phase in train test; do
    cache_dir="${CACHE_ROOT}/${method}/sind/${phase}"
    if [ ! -f "${cache_dir}/file_list.pkl" ]; then
      echo "[error] Missing ${cache_dir}/file_list.pkl" >&2
      exit 1
    fi
    if ! compgen -G "${cache_dir}/*.h5" > /dev/null; then
      echo "[error] Missing h5 cache files in ${cache_dir}" >&2
      exit 1
    fi
  done
done

echo "[done] canonical scenarios, splits, and UniTraj caches were built successfully"
