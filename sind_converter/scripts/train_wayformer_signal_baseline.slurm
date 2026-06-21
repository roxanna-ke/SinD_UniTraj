#!/bin/bash
#SBATCH --job-name=sind_wayformer_signal
#SBATCH --output=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SinD_UniTraj_signal/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:2

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/ke/projects/SinD_UniTraj_signal}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache_signal}"
CACHE_PATH="${CACHE_PATH:-${SCRATCH_ROOT}/unitraj_cache/wayformer}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRATCH_ROOT}/splits/record_level/train/sind}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${SCRATCH_ROOT}/splits/record_level/test/sind}"
EXP_NAME="${EXP_NAME:-sind_wayformer_signal_baseline}"
WANDB_PROJECT="${WANDB_PROJECT:-SinD_UniTraj}"
MAX_DATA_NUM="${MAX_DATA_NUM:-null}"
MAX_VAL_DATA_NUM="${MAX_VAL_DATA_NUM:-null}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
LOAD_NUM_WORKERS="${LOAD_NUM_WORKERS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.05}"
FULL_VALIDATE_LIMIT_VAL_BATCHES="${FULL_VALIDATE_LIMIT_VAL_BATCHES:-1.0}"
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

for phase in train test; do
  if [ ! -f "${CACHE_PATH}/sind/${phase}/file_list.pkl" ]; then
    echo "[error] Wayformer ${phase} cache not found: ${CACHE_PATH}/sind/${phase}/file_list.pkl" >&2
    exit 1
  fi
done

mkdir -p "${PROJECT_ROOT}/logs"

module purge
module load gcc python
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_ROOT}/UniTraj"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PROJECT_ROOT}/scenarionet:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[info] host=$(hostname)"
echo "[info] exp_name=${EXP_NAME}"
echo "[info] wandb_project=${WANDB_PROJECT}"
echo "[info] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[info] cache_path=${CACHE_PATH}"
echo "[info] train_data_path=${TRAIN_DATA_PATH}"
echo "[info] val_data_path=${VAL_DATA_PATH}"
echo "[info] max_data_num=${MAX_DATA_NUM}"
echo "[info] max_val_data_num=${MAX_VAL_DATA_NUM}"
echo "[info] max_epochs=${MAX_EPOCHS}"
echo "[info] load_num_workers=${LOAD_NUM_WORKERS}"
echo "[info] train_batch_size=${TRAIN_BATCH_SIZE}"
echo "[info] eval_batch_size=${EVAL_BATCH_SIZE}"
echo "[info] limit_val_batches=${LIMIT_VAL_BATCHES}"
echo "[info] full_validate_limit_val_batches=${FULL_VALIDATE_LIMIT_VAL_BATCHES}"
echo "[info] devices=${DEVICES}"
python --version
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); raise SystemExit(0 if torch.cuda.device_count() >= 2 else 'Expected at least 2 visible CUDA devices for Wayformer DDP')"

start_ts=$(date +%s)

srun python unitraj/train.py \
  method=wayformer \
  debug=False \
  exp_name="${EXP_NAME}" \
  wandb_project="${WANDB_PROJECT}" \
  "devices=${DEVICES}" \
  use_cache=True \
  overwrite_cache=False \
  use_traffic_light_tokens=True \
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
  limit_val_batches="${LIMIT_VAL_BATCHES}" \
  num_sanity_val_steps="${NUM_SANITY_VAL_STEPS}" \
  full_validate_after_fit=True \
  full_validate_limit_val_batches="${FULL_VALIDATE_LIMIT_VAL_BATCHES}"

end_ts=$(date +%s)
echo "[done] Wayformer signal baseline training and full validation completed in $((end_ts - start_ts)) seconds"
