#!/bin/bash
#SBATCH --job-name=sind_signal_smoke
#SBATCH --output=/home/%u/projects/SinD_UniTraj/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/ke/projects/SinD_UniTraj}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
DATA_ROOT="${DATA_ROOT:-/scratch/izar/ke/sind_raw}"
MAP_FALLBACK_ROOT="${MAP_FALLBACK_ROOT:-${DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache_signal}"
SMOKE_ROOT="${SMOKE_ROOT:-${SCRATCH_ROOT}/smoke_test}"
SMOKE_CANONICAL_ROOT="${SMOKE_CANONICAL_ROOT:-${SMOKE_ROOT}/canonical_scenarios}"
SMOKE_SPLIT_ROOT="${SMOKE_SPLIT_ROOT:-${SMOKE_ROOT}/splits}"
SMOKE_CACHE_ROOT="${SMOKE_CACHE_ROOT:-${SMOKE_ROOT}/unitraj_cache}"
STRIDE="${STRIDE:-40}"
MAX_RECORDS="${MAX_RECORDS:-4}"
MAX_SCENARIOS_PER_RECORD="${MAX_SCENARIOS_PER_RECORD:-2}"
DRY_RUN_SAMPLES="${DRY_RUN_SAMPLES:-4}"
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
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

echo "[info] host=$(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] training_mapping_table=${TRAINING_MAPPING_TABLE}"
echo "[info] smoke_canonical_root=${SMOKE_CANONICAL_ROOT}"
echo "[info] smoke_split_root=${SMOKE_SPLIT_ROOT}"
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
]
for name in repo_imports:
    importlib.import_module(name)

print("Environment and signal-pipeline import smoke check passed.")
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

echo "[step] minimal canonical conversion smoke test"
srun python -m sind_converter.scripts.sind_pipeline convert-scenarios \
  "${COMMON_ARGS[@]}" \
  --max-records "${MAX_RECORDS}" \
  --max-scenarios-per-record "${MAX_SCENARIOS_PER_RECORD}"

canonical_dir="${SMOKE_CANONICAL_ROOT}/sind"
for file_name in dataset_summary.pkl dataset_mapping.pkl; do
  if [ ! -f "${canonical_dir}/${file_name}" ]; then
    echo "[error] Missing ${canonical_dir}/${file_name}" >&2
    exit 1
  fi
done

echo "[step] minimal split smoke test"
srun python -m sind_converter.scripts.sind_pipeline make-splits \
  "${COMMON_ARGS[@]}" \
  --canonical-root "${canonical_dir}" \
  --mode record-level \
  --train-ratio 0.8 \
  --seed 42

for split in train test; do
  split_dir="${SMOKE_SPLIT_ROOT}/record_level/${split}/sind"
  if [ ! -f "${split_dir}/dataset_summary.pkl" ]; then
    echo "[error] Missing ${split_dir}/dataset_summary.pkl" >&2
    exit 1
  fi
  if [ ! -f "${split_dir}/dataset_mapping.pkl" ]; then
    echo "[error] Missing ${split_dir}/dataset_mapping.pkl" >&2
    exit 1
  fi
done

echo "[step] minimal UniTraj validation smoke test"
srun python -m sind_converter.scripts.build_sind_unitraj_caches \
  --train-dir "${SMOKE_SPLIT_ROOT}/record_level/train/sind" \
  --val-dir "${SMOKE_SPLIT_ROOT}/record_level/test/sind" \
  --cache-root "${SMOKE_CACHE_ROOT}" \
  --methods MTR \
  --check-samples 1 \
  --dry-run-samples "${DRY_RUN_SAMPLES}" \
  --validate-only

echo "[done] signal smoke test passed"
