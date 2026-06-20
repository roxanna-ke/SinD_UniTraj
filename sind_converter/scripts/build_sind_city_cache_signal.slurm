#!/bin/bash
#SBATCH --job-name=sind_cityholdout_signal_cache
#SBATCH --output=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/ke/projects/SinD_UniTraj_signal}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-/scratch/izar/ke/sind_raw}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache_signal}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${SCRATCH_ROOT}/canonical_scenarios}"
CITY_HOLDOUT_TAG="${CITY_HOLDOUT_TAG:-xian_holdout_signal}"
SPLIT_ROOT="${SPLIT_ROOT:-${SCRATCH_ROOT}/splits_${CITY_HOLDOUT_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/unitraj_cache_${CITY_HOLDOUT_TAG}}"
TEST_CITY="${TEST_CITY:-Xi_an}"
TRAIN_CITIES="${TRAIN_CITIES:-}"
CHECK_SAMPLES="${CHECK_SAMPLES:-8}"
DRY_RUN_SAMPLES="${DRY_RUN_SAMPLES:-2}"
VERIFY_SCENARIOS="${VERIFY_SCENARIOS:-8}"
METHODS="${METHODS:-MTR wayformer}"
TRAINING_MAPPING_TABLE="${TRAINING_MAPPING_TABLE:-${PROJECT_ROOT}/osm_audit_four_cities/osm_training_mapping_table.csv}"

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

if [ ! -f "${TRAINING_MAPPING_TABLE}" ]; then
  echo "[error] OSM training mapping table not found: ${TRAINING_MAPPING_TABLE}" >&2
  exit 1
fi

CANONICAL_DIR="${CANONICAL_ROOT}/sind"
if [ ! -d "${CANONICAL_DIR}" ]; then
  echo "[error] Canonical ScenarioNet root not found: ${CANONICAL_DIR}" >&2
  echo "[hint] Build signal canonical scenarios first with build_sind_scenarionet_and_caches_signal.slurm or convert-scenarios." >&2
  exit 1
fi

for file_name in dataset_summary.pkl dataset_mapping.pkl; do
  if [ ! -f "${CANONICAL_DIR}/${file_name}" ]; then
    echo "[error] Missing ${CANONICAL_DIR}/${file_name}" >&2
    exit 1
  fi
done

mkdir -p "${SPLIT_ROOT}" "${CACHE_ROOT}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

LOCAL_CACHE_ROOT="${SCRATCH:-$SCRATCH_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_CACHE_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$LOCAL_CACHE_ROOT/torch}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PROJECT_ROOT}/scenarionet:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

echo "[info] host=$(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] canonical_root=${CANONICAL_ROOT}"
echo "[info] split_root=${SPLIT_ROOT}"
echo "[info] cache_root=${CACHE_ROOT}"
echo "[info] test_city=${TEST_CITY}"
echo "[info] train_cities=${TRAIN_CITIES:-<remaining cities>}"
echo "[info] methods=${METHODS}"
python --version
which python

srun python - <<'PY'
import importlib

required = ["omegaconf", "numpy", "h5py", "torch", "metadrive"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages in active environment: {', '.join(missing)}")

repo_imports = [
    "sind_converter.scripts.sind_pipeline",
    "sind_converter.splits.make",
    "sind_converter.lights.bindings",
    "sind_converter.lights.runtime_signal_states",
    "sind_converter.scripts.build_sind_unitraj_caches",
    "sind_converter.scripts.check_signal_cache_contents",
]
for name in repo_imports:
    importlib.import_module(name)

print("Environment and signal city-holdout imports passed.")
PY

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --map-fallback-root "${MAP_FALLBACK_ROOT}"
  --canonical-scenario-root "${CANONICAL_ROOT}"
  --split-root "${SPLIT_ROOT}"
  --cache-root "${CACHE_ROOT}"
  --training-mapping-table "${TRAINING_MAPPING_TABLE}"
)

SPLIT_ARGS=(
  --canonical-root "${CANONICAL_DIR}"
  --mode city-holdout
)

if [ -n "${TRAIN_CITIES}" ]; then
  # shellcheck disable=SC2206
  TRAIN_CITY_ARRAY=(${TRAIN_CITIES})
  SPLIT_ARGS+=(--train-cities "${TRAIN_CITY_ARRAY[@]}" --test-cities "${TEST_CITY}")
else
  SPLIT_ARGS+=(--heldout-cities "${TEST_CITY}")
fi

echo "[step] ScenarioNet city-holdout split generation"
srun python -m sind_converter.scripts.sind_pipeline make-splits \
  "${COMMON_ARGS[@]}" \
  "${SPLIT_ARGS[@]}"

TRAIN_SPLIT_DIR="${SPLIT_ROOT}/city_holdout/train/sind"
VAL_SPLIT_DIR="${SPLIT_ROOT}/city_holdout/test/sind"

for split_dir in "${TRAIN_SPLIT_DIR}" "${VAL_SPLIT_DIR}"; do
  for file_name in dataset_summary.pkl dataset_mapping.pkl; do
    if [ ! -f "${split_dir}/${file_name}" ]; then
      echo "[error] Missing ${split_dir}/${file_name}" >&2
      exit 1
    fi
  done
done

echo "[step] validating city assignment in generated split"
python - <<'PY' "${TRAIN_SPLIT_DIR}" "${VAL_SPLIT_DIR}" "${TEST_CITY}"
import pickle
import sys
from pathlib import Path

train_dir = Path(sys.argv[1])
test_dir = Path(sys.argv[2])
test_city = sys.argv[3]

def load_cities(split_dir: Path) -> set[str]:
    with (split_dir / "dataset_summary.pkl").open("rb") as f:
        summary = pickle.load(f)
    return {str(meta.get("city") or meta.get("split_candidates", {}).get("city")) for meta in summary.values()}

train_cities = load_cities(train_dir)
test_cities = load_cities(test_dir)

if test_cities != {test_city}:
    raise SystemExit(f"[error] test split cities mismatch: expected only {test_city}, got {sorted(test_cities)}")
if test_city in train_cities:
    raise SystemExit(f"[error] holdout city {test_city} leaked into train split")

print(f"[ok] train cities={sorted(train_cities)}")
print(f"[ok] test cities={sorted(test_cities)}")
PY

echo "[step] UniTraj cache validation and build"
# shellcheck disable=SC2206
METHOD_ARRAY=(${METHODS})
srun python -m sind_converter.scripts.build_sind_unitraj_caches \
  --train-dir "${TRAIN_SPLIT_DIR}" \
  --val-dir "${VAL_SPLIT_DIR}" \
  --cache-root "${CACHE_ROOT}" \
  --methods "${METHOD_ARRAY[@]}" \
  --check-samples "${CHECK_SAMPLES}" \
  --dry-run-samples "${DRY_RUN_SAMPLES}" \
  --use-traffic-light-tokens \
  --overwrite

for method in "${METHOD_ARRAY[@]}"; do
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

echo "[step] verify signal info survives into the final cache"
for method in "${METHOD_ARRAY[@]}"; do
  for phase in train test; do
    if [ "${phase}" = "test" ]; then
      split_dir="${VAL_SPLIT_DIR}"
    else
      split_dir="${TRAIN_SPLIT_DIR}"
    fi
    cache_dir="${CACHE_ROOT}/${method}/sind/${phase}"
    srun python -m sind_converter.scripts.check_signal_cache_contents \
      --split-dir "${split_dir}" \
      --cache-dir "${cache_dir}" \
      --sample-count "${VERIFY_SCENARIOS}" \
      --require-light-tokens
  done
done

echo "[done] signal city-holdout split and caches were built successfully"
