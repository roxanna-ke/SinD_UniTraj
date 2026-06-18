#!/bin/bash
#SBATCH --job-name=sind_mtr_smoke
#SBATCH --output=/home/%u/projects/SinD_UniTraj/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/ke/projects/SinD_UniTraj}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"
CACHE_PATH="${CACHE_PATH:-${SCRATCH_ROOT}/unitraj_cache/MTR}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRATCH_ROOT}/splits/record_level/train/sind}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${SCRATCH_ROOT}/splits/record_level/test/sind}"
EXP_NAME="${EXP_NAME:-sind_MTR_smoke}"
MAX_DATA_NUM="${MAX_DATA_NUM:-2048}"
MAX_VAL_DATA_NUM="${MAX_VAL_DATA_NUM:-1024}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LOAD_NUM_WORKERS="${LOAD_NUM_WORKERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-8}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-4}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
DEVICES="${DEVICES:-[0,1]}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[error] Virtual environment not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/UniTraj" ]; then
  echo "[error] UniTraj root not found: ${PROJECT_ROOT}/UniTraj" >&2
  exit 1
fi

if [ ! -f "${CACHE_PATH}/sind/train/file_list.pkl" ]; then
  echo "[error] MTR train cache not found: ${CACHE_PATH}/sind/train/file_list.pkl" >&2
  exit 1
fi

if [ ! -f "${CACHE_PATH}/sind/test/file_list.pkl" ]; then
  echo "[error] MTR test cache not found: ${CACHE_PATH}/sind/test/file_list.pkl" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_ROOT}/UniTraj"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PROJECT_ROOT}/scenarionet:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[info] host=$(hostname)"
echo "[info] project_root=${PROJECT_ROOT}"
echo "[info] cache_path=${CACHE_PATH}"
echo "[info] train_data_path=${TRAIN_DATA_PATH}"
echo "[info] val_data_path=${VAL_DATA_PATH}"
echo "[info] exp_name=${EXP_NAME}"
echo "[info] max_data_num=${MAX_DATA_NUM}"
echo "[info] max_val_data_num=${MAX_VAL_DATA_NUM}"
echo "[info] max_epochs=${MAX_EPOCHS}"
echo "[info] load_num_workers=${LOAD_NUM_WORKERS}"
echo "[info] train_batch_size=${TRAIN_BATCH_SIZE}"
echo "[info] eval_batch_size=${EVAL_BATCH_SIZE}"
echo "[info] limit_train_batches=${LIMIT_TRAIN_BATCHES}"
echo "[info] limit_val_batches=${LIMIT_VAL_BATCHES}"
echo "[info] num_sanity_val_steps=${NUM_SANITY_VAL_STEPS}"
python --version
which python

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "from unitraj.models.mtr.ops.attention import attention_cuda; from unitraj.models.mtr.ops.knn import knn_cuda; print('MTR CUDA ops import OK')"

start_ts=$(date +%s)

python unitraj/train.py \
  method=MTR \
  debug=False \
  exp_name="${EXP_NAME}" \
  "devices=${DEVICES}" \
  use_cache=True \
  overwrite_cache=False \
  cache_path="${CACHE_PATH}" \
  "train_data_path=[${TRAIN_DATA_PATH}]" \
  "val_data_path=[${VAL_DATA_PATH}]" \
  "max_data_num=[${MAX_DATA_NUM}]" \
  max_val_data_num="${MAX_VAL_DATA_NUM}" \
  "starting_frame=[0]" \
  load_num_workers="${LOAD_NUM_WORKERS}" \
  method.max_epochs="${MAX_EPOCHS}" \
  method.train_batch_size="${TRAIN_BATCH_SIZE}" \
  method.eval_batch_size="${EVAL_BATCH_SIZE}" \
  limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
  limit_val_batches="${LIMIT_VAL_BATCHES}" \
  num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}"

end_ts=$(date +%s)
echo "[done] MTR smoke training completed in $((end_ts - start_ts)) seconds"
