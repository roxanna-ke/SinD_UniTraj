#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"

METHOD="${METHOD:-MTR}"
CKPT_PATH="${CKPT_PATH:-}"
EXP_NAME="${EXP_NAME:-sind_${METHOD}_eval}"
WANDB_PROJECT="${WANDB_PROJECT:-SinD_UniTraj}"

SPLIT_MODE="${SPLIT_MODE:-record_level}"
SIGNAL="${SIGNAL:-false}"
USE_TRAFFIC_LIGHT_TOKENS="${USE_TRAFFIC_LIGHT_TOKENS:-${SIGNAL}}"

if [ "${SPLIT_MODE}" = "city_holdout" ]; then
  CITY_HOLDOUT_TAG="${CITY_HOLDOUT_TAG:-xian_holdout_signal}"
  SPLIT_ROOT="${SPLIT_ROOT:-${SCRATCH_ROOT}/splits_${CITY_HOLDOUT_TAG}}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/unitraj_cache_${CITY_HOLDOUT_TAG}}"
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SPLIT_ROOT}/city_holdout/train/sind}"
  VAL_DATA_PATH="${VAL_DATA_PATH:-${SPLIT_ROOT}/city_holdout/test/sind}"
else
  SPLIT_ROOT="${SPLIT_ROOT:-${SCRATCH_ROOT}/splits}"
  CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/unitraj_cache}"
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SPLIT_ROOT}/record_level/train/sind}"
  VAL_DATA_PATH="${VAL_DATA_PATH:-${SPLIT_ROOT}/record_level/test/sind}"
fi

CACHE_METHOD="${CACHE_METHOD:-${METHOD}}"
if [ "${CACHE_METHOD}" = "wayformer" ]; then
  CACHE_METHOD="wayformer"
elif [ "${CACHE_METHOD}" = "Wayformer" ]; then
  CACHE_METHOD="wayformer"
fi
CACHE_PATH="${CACHE_PATH:-${CACHE_ROOT}/${CACHE_METHOD}}"

MAX_DATA_NUM="${MAX_DATA_NUM:-null}"
MAX_VAL_DATA_NUM="${MAX_VAL_DATA_NUM:-null}"
LOAD_NUM_WORKERS="${LOAD_NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
DEVICES="${DEVICES:-[0]}"
DEBUG="${DEBUG:-False}"
NUM_IMAGES="${NUM_IMAGES:-32}"
VIS_BATCH_SIZE="${VIS_BATCH_SIZE:-16}"
VIS_DEVICE="${VIS_DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/prediction_visualizations}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
RUN_VISUALIZATION="${RUN_VISUALIZATION:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if [ -z "${CKPT_PATH}" ]; then
  echo "[error] CKPT_PATH is required" >&2
  exit 1
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/UniTraj" ]; then
  echo "[error] UniTraj root not found: ${PROJECT_ROOT}/UniTraj" >&2
  exit 1
fi

if [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "[warn] virtual environment not found, using current Python: ${VENV_DIR}/bin/activate" >&2
fi

for path in "${TRAIN_DATA_PATH}" "${VAL_DATA_PATH}"; do
  if [ ! -f "${path}/dataset_summary.pkl" ]; then
    echo "[error] missing ScenarioNet split: ${path}/dataset_summary.pkl" >&2
    exit 1
  fi
done

if [ ! -f "${CACHE_PATH}/sind/test/file_list.pkl" ]; then
  echo "[error] UniTraj test cache not found: ${CACHE_PATH}/sind/test/file_list.pkl" >&2
  exit 1
fi

mkdir -p "${VIS_OUTPUT_DIR}"

cd "${PROJECT_ROOT}/UniTraj"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PROJECT_ROOT}/scenarionet:${PYTHONPATH:-}"
export WANDB_PROJECT
export WANDB_MODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

COMMON_OVERRIDES=(
  "method=${METHOD}"
  "debug=${DEBUG}"
  "exp_name=${EXP_NAME}"
  "wandb_project=${WANDB_PROJECT}"
  "devices=${DEVICES}"
  "ckpt_path=${CKPT_PATH}"
  "use_cache=True"
  "overwrite_cache=False"
  "use_traffic_light_tokens=${USE_TRAFFIC_LIGHT_TOKENS}"
  "cache_path=${CACHE_PATH}"
  "train_data_path=[${TRAIN_DATA_PATH}]"
  "val_data_path=[${VAL_DATA_PATH}]"
  "max_data_num=[${MAX_DATA_NUM}]"
  "max_val_data_num=${MAX_VAL_DATA_NUM}"
  "starting_frame=[0]"
  "load_num_workers=${LOAD_NUM_WORKERS}"
  "method.eval_batch_size=${EVAL_BATCH_SIZE}"
)

echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] method=${METHOD}"
echo "[info] exp_name=${EXP_NAME}"
echo "[info] ckpt_path=${CKPT_PATH}"
echo "[info] split_mode=${SPLIT_MODE}"
echo "[info] cache_path=${CACHE_PATH}"
echo "[info] val_data_path=${VAL_DATA_PATH}"
echo "[info] use_traffic_light_tokens=${USE_TRAFFIC_LIGHT_TOKENS}"
echo "[info] visualization_output_dir=${VIS_OUTPUT_DIR}"

if [ "${RUN_EVALUATION}" = "true" ]; then
  echo "[step] evaluation"
  python unitraj/evaluation.py "${COMMON_OVERRIDES[@]}"
fi

if [ "${RUN_VISUALIZATION}" = "true" ]; then
  echo "[step] prediction visualization"
  python unitraj/visualize_predictions.py \
    "${COMMON_OVERRIDES[@]}" \
    "visualization_output_dir=${VIS_OUTPUT_DIR}" \
    "num_prediction_visualizations=${NUM_IMAGES}" \
    "visualization_batch_size=${VIS_BATCH_SIZE}" \
    "visualization_device=${VIS_DEVICE}"
fi

echo "[done] outputs are in ${VIS_OUTPUT_DIR}"
