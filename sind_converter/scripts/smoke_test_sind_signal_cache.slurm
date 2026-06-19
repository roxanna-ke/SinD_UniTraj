#!/bin/bash
#SBATCH --job-name=sind_signal_cache_smoke
#SBATCH --output=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --account=master

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/ke/projects/SinD_UniTraj_signal}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-/scratch/izar/ke/sind_raw}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache_signal}"
SMOKE_ROOT="${SMOKE_ROOT:-${SCRATCH_ROOT}/smoke_test_cache}"
SMOKE_CANONICAL_ROOT="${SMOKE_CANONICAL_ROOT:-${SMOKE_ROOT}/canonical_scenarios}"
SMOKE_SPLIT_ROOT="${SMOKE_SPLIT_ROOT:-${SMOKE_ROOT}/splits}"
SMOKE_CACHE_ROOT="${SMOKE_CACHE_ROOT:-${SMOKE_ROOT}/unitraj_cache}"
STRIDE="${STRIDE:-40}"
MAX_RECORDS="${MAX_RECORDS:-4}"
MAX_SCENARIOS_PER_RECORD="${MAX_SCENARIOS_PER_RECORD:-2}"
CHECK_SAMPLES="${CHECK_SAMPLES:-2}"
DRY_RUN_SAMPLES="${DRY_RUN_SAMPLES:-4}"
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
  echo "[hint] Run sind_pipeline.py audit-maps first, or set TRAINING_MAPPING_TABLE to the audited CSV." >&2
  exit 1
fi

mkdir -p "${SMOKE_CANONICAL_ROOT}" "${SMOKE_SPLIT_ROOT}" "${SMOKE_CACHE_ROOT}" "${PROJECT_ROOT}/logs"

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
echo "[info] training_mapping_table=${TRAINING_MAPPING_TABLE}"
echo "[info] smoke_canonical_root=${SMOKE_CANONICAL_ROOT}"
echo "[info] smoke_split_root=${SMOKE_SPLIT_ROOT}"
echo "[info] smoke_cache_root=${SMOKE_CACHE_ROOT}"
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
    "sind_converter.data.discovery",
    "sind_converter.scripts.sind_pipeline",
    "sind_converter.scenarios.convert",
    "sind_converter.splits.make",
    "sind_converter.lights.bindings",
    "sind_converter.lights.runtime_signal_states",
    "sind_converter.scripts.build_sind_unitraj_caches",
    "sind_converter.scripts.check_signal_cache_contents",
]
for name in repo_imports:
    importlib.import_module(name)

print("Environment and signal-cache smoke imports passed.")
PY

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --map-fallback-root "${MAP_FALLBACK_ROOT}"
  --canonical-scenario-root "${SMOKE_CANONICAL_ROOT}"
  --split-root "${SMOKE_SPLIT_ROOT}"
  --cache-root "${SMOKE_CACHE_ROOT}"
  --training-mapping-table "${TRAINING_MAPPING_TABLE}"
  --stride "${STRIDE}"
)

echo "[step] build minimal canonical scenarios with traffic lights"
srun python -m sind_converter.scripts.sind_pipeline convert-scenarios \
  "${COMMON_ARGS[@]}" \
  --max-records "${MAX_RECORDS}" \
  --max-scenarios-per-record "${MAX_SCENARIOS_PER_RECORD}"

CANONICAL_DIR="${SMOKE_CANONICAL_ROOT}/sind"
for file_name in dataset_summary.pkl dataset_mapping.pkl; do
  if [ ! -f "${CANONICAL_DIR}/${file_name}" ]; then
    echo "[error] Missing ${CANONICAL_DIR}/${file_name}" >&2
    exit 1
  fi
done

echo "[step] build minimal record-level split"
srun python -m sind_converter.scripts.sind_pipeline make-splits \
  "${COMMON_ARGS[@]}" \
  --canonical-root "${CANONICAL_DIR}" \
  --mode record-level \
  --train-ratio 0.8 \
  --seed 42

TRAIN_SPLIT_DIR="${SMOKE_SPLIT_ROOT}/record_level/train/sind"
VAL_SPLIT_DIR="${SMOKE_SPLIT_ROOT}/record_level/test/sind"
for split_dir in "${TRAIN_SPLIT_DIR}" "${VAL_SPLIT_DIR}"; do
  for file_name in dataset_summary.pkl dataset_mapping.pkl; do
    if [ ! -f "${split_dir}/${file_name}" ]; then
      echo "[error] Missing ${split_dir}/${file_name}" >&2
      exit 1
    fi
  done
done

echo "[step] build final UniTraj cache"
# shellcheck disable=SC2206
METHOD_ARRAY=(${METHODS})
srun python -m sind_converter.scripts.build_sind_unitraj_caches \
  --train-dir "${TRAIN_SPLIT_DIR}" \
  --val-dir "${VAL_SPLIT_DIR}" \
  --cache-root "${SMOKE_CACHE_ROOT}" \
  --methods "${METHOD_ARRAY[@]}" \
  --check-samples "${CHECK_SAMPLES}" \
  --dry-run-samples "${DRY_RUN_SAMPLES}" \
  --use-traffic-light-tokens \
  --overwrite

for method in "${METHOD_ARRAY[@]}"; do
  for phase in train test; do
    cache_dir="${SMOKE_CACHE_ROOT}/${method}/sind/${phase}"
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
    split_dir="${SMOKE_SPLIT_ROOT}/record_level/${phase}/sind"
    if [ "${phase}" = "test" ]; then
      split_dir="${VAL_SPLIT_DIR}"
    fi
    cache_dir="${SMOKE_CACHE_ROOT}/${method}/sind/${phase}"
    srun python -m sind_converter.scripts.check_signal_cache_contents \
      --split-dir "${split_dir}" \
      --cache-dir "${cache_dir}" \
      --sample-count "${VERIFY_SCENARIOS}" \
      --require-light-tokens
  done
done

echo "[done] scenario generation, cache build, and signal retention checks all passed"
