#!/bin/bash
#SBATCH --job-name=sind_visualization
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

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mtr-izar}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"

METHOD="${METHOD:-MTR}"
CKPT_PATH="${CKPT_PATH:-}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/UniTraj/unitraj_ckpt}"
EXP_NAME="${EXP_NAME:-sind_${METHOD}_eval}"
WANDB_PROJECT="${WANDB_PROJECT:-SinD_UniTraj}"

SPLIT_MODE="${SPLIT_MODE:-record_level}"
SIGNAL="${SIGNAL:-false}"
USE_TRAFFIC_LIGHT_TOKENS="${USE_TRAFFIC_LIGHT_TOKENS:-${SIGNAL}}"
RUN_SUITE="${RUN_SUITE:-true}"
SUITE_INCLUDE="${SUITE_INCLUDE:-mtr_baseline,mtr_signal,wayformer_baseline,wayformer_signal,mtr_cityholdout,mtr_signal_cityholdout,wayformer_cityholdout,wayformer_signal_cityholdout}"
BASELINE_SCRATCH_ROOT="${BASELINE_SCRATCH_ROOT:-/scratch/izar/ke/sind_cache}"
SIGNAL_SCRATCH_ROOT="${SIGNAL_SCRATCH_ROOT:-/scratch/izar/ke/sind_cache_signal}"
BASELINE_CITY_HOLDOUT_TAG="${BASELINE_CITY_HOLDOUT_TAG:-xian_holdout}"
SIGNAL_CITY_HOLDOUT_TAG="${SIGNAL_CITY_HOLDOUT_TAG:-xian_holdout_signal}"
CITY_HOLDOUT_NAME="${CITY_HOLDOUT_NAME:-Xi_an}"
CITY_HOLDOUT_NAMES="${CITY_HOLDOUT_NAMES:-Xi_an Changchun Chongqing Tianjin}"
CITY_HOLDOUT_CKPT_CITIES="${CITY_HOLDOUT_CKPT_CITIES:-Xi_an}"

MAX_DATA_NUM="${MAX_DATA_NUM:-null}"
MAX_VAL_DATA_NUM="${MAX_VAL_DATA_NUM:-256}"
LOAD_NUM_WORKERS="${LOAD_NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
DEVICES="${DEVICES:-[0]}"
DEBUG="${DEBUG:-False}"
NUM_IMAGES="${NUM_IMAGES:-2048}"
VIS_BATCH_SIZE="${VIS_BATCH_SIZE:-8}"
VIS_DEVICE="${VIS_DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/prediction_visualizations}"
USER_VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${OUTPUT_ROOT}/${EXP_NAME}}"
AGGREGATE_VISUALIZATION="${AGGREGATE_VISUALIZATION:-true}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-true}"
AGGREGATE_MAX_TRACKS="${AGGREGATE_MAX_TRACKS:-3}"
AGGREGATE_MIN_TRACK_DISTANCE="${AGGREGATE_MIN_TRACK_DISTANCE:-4.0}"
AGGREGATE_MIN_TOTAL_STEPS="${AGGREGATE_MIN_TOTAL_STEPS:-61}"
VISUALIZATION_DATA_ROOT="${VISUALIZATION_DATA_ROOT:-/scratch/izar/ke/sind_raw}"
VISUALIZATION_MAP_FALLBACK_ROOT="${VISUALIZATION_MAP_FALLBACK_ROOT:-${VISUALIZATION_DATA_ROOT}}"
RUN_EVALUATION="${RUN_EVALUATION:-false}"
RUN_VISUALIZATION="${RUN_VISUALIZATION:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"

USER_SPLIT_ROOT="${SPLIT_ROOT:-}"
USER_CACHE_ROOT="${CACHE_ROOT:-}"
USER_TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
USER_VAL_DATA_PATH="${VAL_DATA_PATH:-}"
USER_CACHE_PATH="${CACHE_PATH:-}"
USER_CACHE_METHOD="${CACHE_METHOD:-}"
USER_CITY_HOLDOUT_TAG="${CITY_HOLDOUT_TAG:-}"

if [ ! -d "${PROJECT_ROOT}/UniTraj" ]; then
  echo "[error] UniTraj root not found: ${PROJECT_ROOT}/UniTraj" >&2
  exit 1
fi

if [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "[warn] virtual environment not found, using current Python: ${VENV_DIR}/bin/activate" >&2
fi

cd "${PROJECT_ROOT}/UniTraj"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/UniTraj:${PROJECT_ROOT}/scenarionet:${PYTHONPATH:-}"
export WANDB_PROJECT
export WANDB_MODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

normalize_cache_method() {
  local method="$1"
  if [ "${method}" = "wayformer" ] || [ "${method}" = "Wayformer" ]; then
    echo "wayformer"
  else
    echo "${method}"
  fi
}

resolve_checkpoint() {
  local candidate="$1"
  python - "$candidate" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_file():
    print(path)
    raise SystemExit(0)
if not path.is_dir():
    raise SystemExit(f"[error] checkpoint path not found: {path}")

files = [p for p in path.rglob("*") if p.is_file()]
if not files:
    raise SystemExit(f"[error] no checkpoint files found under: {path}")

def rank(p: Path):
    match = re.search(r"epoch[=_-](\d+)", p.name)
    epoch = int(match.group(1)) if match else -1
    ckpt_bonus = 1 if p.suffix == ".ckpt" else 0
    return (epoch, ckpt_bonus, p.stat().st_mtime, str(p))

print(max(files, key=rank))
PY
}

configure_paths() {
  local scratch_root="$1"
  local method="$2"
  local split_mode="$3"
  local city_holdout_tag="$4"
  local cache_method
  cache_method="$(normalize_cache_method "${CACHE_METHOD:-${method}}")"

  if [ "${split_mode}" = "city_holdout" ]; then
    CITY_HOLDOUT_TAG="${CITY_HOLDOUT_TAG:-${city_holdout_tag}}"
    SPLIT_ROOT="${SPLIT_ROOT:-${scratch_root}/splits_${CITY_HOLDOUT_TAG}}"
    CACHE_ROOT="${CACHE_ROOT:-${scratch_root}/unitraj_cache_${CITY_HOLDOUT_TAG}}"
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SPLIT_ROOT}/city_holdout/train/sind}"
    VAL_DATA_PATH="${VAL_DATA_PATH:-${SPLIT_ROOT}/city_holdout/test/sind}"
  else
    SPLIT_ROOT="${SPLIT_ROOT:-${scratch_root}/splits}"
    CACHE_ROOT="${CACHE_ROOT:-${scratch_root}/unitraj_cache}"
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SPLIT_ROOT}/record_level/train/sind}"
    VAL_DATA_PATH="${VAL_DATA_PATH:-${SPLIT_ROOT}/record_level/test/sind}"
  fi
  CACHE_PATH="${CACHE_PATH:-${CACHE_ROOT}/${cache_method}}"
}

count_cache_samples() {
  local file_list_path="$1"
  python - "$file_list_path" "${MAX_VAL_DATA_NUM}" <<'PY'
import pickle
import sys
from pathlib import Path

path = Path(sys.argv[1])
limit_raw = sys.argv[2]
with path.open("rb") as handle:
    file_list = pickle.load(handle)
total = len(file_list)
limit = None if limit_raw in {"", "null", "None"} else int(limit_raw)
effective = total if limit is None else min(total, limit)
print(f"{effective}/{total}")
PY
}

run_one() {
  local label="$1"
  local method="$2"
  local ckpt_candidate="$3"
  local exp_name="$4"
  local signal="$5"
  local scratch_root="$6"
  local lane_control_map_tokens="$7"
  local split_mode="$8"
  local city_holdout_tag="$9"
  local expected_aggregate_cities="${10:-}"
  local aggregate_cities_arg="${expected_aggregate_cities:-${CITY_HOLDOUT_NAMES}}"

  local ckpt_path
  ckpt_path="$(resolve_checkpoint "${ckpt_candidate}")"

  unset SPLIT_ROOT CACHE_ROOT TRAIN_DATA_PATH VAL_DATA_PATH CACHE_PATH CACHE_METHOD CITY_HOLDOUT_TAG
  if [ -n "${USER_SPLIT_ROOT}" ]; then SPLIT_ROOT="${USER_SPLIT_ROOT}"; fi
  if [ -n "${USER_CACHE_ROOT}" ]; then CACHE_ROOT="${USER_CACHE_ROOT}"; fi
  if [ -n "${USER_TRAIN_DATA_PATH}" ]; then TRAIN_DATA_PATH="${USER_TRAIN_DATA_PATH}"; fi
  if [ -n "${USER_VAL_DATA_PATH}" ]; then VAL_DATA_PATH="${USER_VAL_DATA_PATH}"; fi
  if [ -n "${USER_CACHE_PATH}" ]; then CACHE_PATH="${USER_CACHE_PATH}"; fi
  if [ -n "${USER_CACHE_METHOD}" ]; then CACHE_METHOD="${USER_CACHE_METHOD}"; fi
  if [ -n "${USER_CITY_HOLDOUT_TAG}" ]; then CITY_HOLDOUT_TAG="${USER_CITY_HOLDOUT_TAG}"; fi
  configure_paths "${scratch_root}" "${method}" "${split_mode}" "${city_holdout_tag}"

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

  local output_dir="${OUTPUT_ROOT}/${exp_name}"
  if [ "${RUN_SUITE}" != "true" ] && [ -n "${USER_VIS_OUTPUT_DIR}" ]; then
    output_dir="${USER_VIS_OUTPUT_DIR}"
  fi
  mkdir -p "${output_dir}"
  if [ "${RUN_VISUALIZATION}" = "true" ]; then
    find "${output_dir}" -maxdepth 1 -type f -name "*.png" -delete
  fi

  local sample_count
  sample_count="$(count_cache_samples "${CACHE_PATH}/sind/test/file_list.pkl")"

  local common_overrides=(
    "method=${method}"
    "debug=${DEBUG}"
    "exp_name=${exp_name}"
    "wandb_project=${WANDB_PROJECT}"
    "devices=${DEVICES}"
    "ckpt_path='${ckpt_path}'"
    "use_cache=True"
    "overwrite_cache=False"
    "use_traffic_light_tokens=${signal}"
    "use_lane_control_state_in_map_tokens=${lane_control_map_tokens}"
    "cache_path=${CACHE_PATH}"
    "train_data_path=[${TRAIN_DATA_PATH}]"
    "val_data_path=[${VAL_DATA_PATH}]"
    "max_data_num=[${MAX_DATA_NUM}]"
    "max_val_data_num=${MAX_VAL_DATA_NUM}"
    "starting_frame=[0]"
    "load_num_workers=${LOAD_NUM_WORKERS}"
    "method.eval_batch_size=${EVAL_BATCH_SIZE}"
  )

  if [ "${lane_control_map_tokens}" = "true" ]; then
    if [ "${method}" = "MTR" ]; then
      common_overrides+=("method.CONTEXT_ENCODER.NUM_INPUT_ATTR_MAP=38")
    elif [ "${method}" = "wayformer" ] || [ "${method}" = "Wayformer" ]; then
      common_overrides+=("method.num_map_feature=38")
    fi
  else
    if [ "${method}" = "MTR" ]; then
      common_overrides+=("method.CONTEXT_ENCODER.NUM_INPUT_ATTR_MAP=29")
    elif [ "${method}" = "wayformer" ] || [ "${method}" = "Wayformer" ]; then
      common_overrides+=("method.num_map_feature=29")
    fi
  fi

  echo "[info] label=${label}"
  echo "[info] project_root=${PROJECT_ROOT}"
  echo "[info] method=${method}"
  echo "[info] exp_name=${exp_name}"
  echo "[info] ckpt_path=${ckpt_path}"
  echo "[info] split_mode=${split_mode}"
  if [ "${split_mode}" = "city_holdout" ]; then
    echo "[info] city_holdout_tag=${CITY_HOLDOUT_TAG}"
  fi
  echo "[info] cache_path=${CACHE_PATH}"
  echo "[info] val_data_path=${VAL_DATA_PATH}"
  echo "[info] validation_samples=${sample_count} after MAX_VAL_DATA_NUM/total"
  echo "[info] num_prediction_visualizations=${NUM_IMAGES}"
  echo "[info] use_traffic_light_tokens=${signal}"
  echo "[info] use_lane_control_state_in_map_tokens=${lane_control_map_tokens}"
  echo "[info] visualization_output_dir=${output_dir}"

  if [ "${RUN_EVALUATION}" = "true" ]; then
    echo "[step] evaluation: ${label}"
    python unitraj/evaluation.py "${common_overrides[@]}"
  fi

  if [ "${RUN_VISUALIZATION}" = "true" ]; then
    echo "[step] prediction visualization: ${label}"
    python unitraj/visualize_predictions.py \
      "${common_overrides[@]}" \
      "+visualization_output_dir=${output_dir}" \
      "+num_prediction_visualizations=${NUM_IMAGES}" \
      "+visualization_batch_size=${VIS_BATCH_SIZE}" \
      "+visualization_device=${VIS_DEVICE}" \
      "+aggregate_visualization=${AGGREGATE_VISUALIZATION}" \
      "+aggregate_only=${AGGREGATE_ONLY}" \
      "+aggregate_max_tracks=${AGGREGATE_MAX_TRACKS}" \
      "+aggregate_min_track_distance=${AGGREGATE_MIN_TRACK_DISTANCE}" \
      "+aggregate_min_total_steps=${AGGREGATE_MIN_TOTAL_STEPS}" \
      "+aggregate_cities=${aggregate_cities_arg}" \
      "+visualization_data_root=${VISUALIZATION_DATA_ROOT}" \
      "+visualization_map_fallback_root=${VISUALIZATION_MAP_FALLBACK_ROOT}"
    if [ "${AGGREGATE_VISUALIZATION}" = "true" ] && [ -n "${expected_aggregate_cities}" ]; then
      # shellcheck disable=SC2206
      EXPECTED_CITY_ARRAY=(${expected_aggregate_cities})
      for expected_city in "${EXPECTED_CITY_ARRAY[@]}"; do
        aggregate_path="${output_dir}/aggregate_${expected_city}.png"
        if [ ! -s "${aggregate_path}" ]; then
          echo "[error] missing expected aggregate visualization: ${aggregate_path}" >&2
          exit 1
        fi
      done
    fi
  fi

  echo "[done] ${label} outputs are in ${output_dir}"
}

suite_contains() {
  local needle="$1"
  case ",${SUITE_INCLUDE}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}

city_tag_prefix() {
  local city="$1"
  case "${city}" in
    Xi_an|Xi\'an|Xian|xi_an|xian) echo "xian" ;;
    Changchun|changchun) echo "changchun" ;;
    Chongqing|chongqing) echo "chongqing" ;;
    Tianjin|tianjin) echo "tianjin" ;;
    *) echo "${city}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//' ;;
  esac
}

city_env_prefix() {
  city_tag_prefix "$1" | tr '[:lower:]' '[:upper:]'
}

baseline_city_holdout_tag() {
  local city="$1"
  local prefix
  prefix="$(city_tag_prefix "${city}")"
  local var_name
  var_name="$(echo "${prefix}_holdout_tag" | tr '[:lower:]' '[:upper:]')"
  local specific="${!var_name:-}"
  echo "${specific:-${prefix}_holdout}"
}

signal_city_holdout_tag() {
  local city="$1"
  local prefix
  prefix="$(city_tag_prefix "${city}")"
  local var_name
  var_name="$(echo "${prefix}_signal_holdout_tag" | tr '[:lower:]' '[:upper:]')"
  local specific="${!var_name:-}"
  echo "${specific:-${prefix}_holdout_signal}"
}

ckpt_candidate_for() {
  local env_prefix="$1"
  local default_path="$2"
  local path_var="${env_prefix}_CKPT_PATH"
  local dir_var="${env_prefix}_CKPT_DIR"
  if [ -n "${!path_var:-}" ]; then
    echo "${!path_var}"
  elif [ -n "${!dir_var:-}" ]; then
    echo "${!dir_var}"
  else
    echo "${default_path}"
  fi
}

if [ "${RUN_SUITE}" = "true" ]; then
  echo "[info] record_level_aggregate_cities=${CITY_HOLDOUT_NAMES}"
  echo "[info] city_holdout_ckpt_cities=${CITY_HOLDOUT_CKPT_CITIES}"
  if suite_contains "mtr_baseline"; then
    run_one "mtr_baseline" "MTR" "$(ckpt_candidate_for MTR_BASELINE "${CKPT_ROOT}/sind_MTR_baseline")" "sind_MTR_baseline_eval" "false" "${BASELINE_SCRATCH_ROOT}" "false" "record_level" "" "${CITY_HOLDOUT_NAMES}"
  fi
  if suite_contains "mtr_signal"; then
    run_one "mtr_signal" "MTR" "$(ckpt_candidate_for MTR_SIGNAL "${CKPT_ROOT}/sind_MTR_signal_baseline")" "sind_MTR_signal_baseline_eval" "true" "${SIGNAL_SCRATCH_ROOT}" "true" "record_level" "" "${CITY_HOLDOUT_NAMES}"
  fi
  if suite_contains "wayformer_baseline"; then
    run_one "wayformer_baseline" "wayformer" "$(ckpt_candidate_for WAYFORMER_BASELINE "${CKPT_ROOT}/sind_wayformer_baseline")" "sind_wayformer_baseline_eval" "false" "${BASELINE_SCRATCH_ROOT}" "false" "record_level" "" "${CITY_HOLDOUT_NAMES}"
  fi
  if suite_contains "wayformer_signal"; then
    run_one "wayformer_signal" "wayformer" "$(ckpt_candidate_for WAYFORMER_SIGNAL "${CKPT_ROOT}/sind_wayformer_signal_baseline")" "sind_wayformer_signal_baseline_eval" "true" "${SIGNAL_SCRATCH_ROOT}" "true" "record_level" "" "${CITY_HOLDOUT_NAMES}"
  fi
  # shellcheck disable=SC2206
  CITY_HOLDOUT_ARRAY=(${CITY_HOLDOUT_CKPT_CITIES})
  for city in "${CITY_HOLDOUT_ARRAY[@]}"; do
    baseline_tag="$(baseline_city_holdout_tag "${city}")"
    signal_tag="$(signal_city_holdout_tag "${city}")"
    city_env="$(city_env_prefix "${city}")"
    if suite_contains "mtr_cityholdout"; then
      run_one "mtr_cityholdout_${city}" "MTR" "$(ckpt_candidate_for "MTR_CITYHOLDOUT_${city_env}" "${CKPT_ROOT}/sind_MTR_cityholdout_${city}")" "sind_MTR_cityholdout_${city}_eval" "false" "${BASELINE_SCRATCH_ROOT}" "false" "city_holdout" "${baseline_tag}" "${city}"
    fi
    if suite_contains "mtr_signal_cityholdout"; then
      run_one "mtr_signal_cityholdout_${city}" "MTR" "$(ckpt_candidate_for "MTR_SIGNAL_CITYHOLDOUT_${city_env}" "${CKPT_ROOT}/sind_MTR_signal_cityholdout_${city}")" "sind_MTR_signal_cityholdout_${city}_eval" "true" "${SIGNAL_SCRATCH_ROOT}" "true" "city_holdout" "${signal_tag}" "${city}"
    fi
    if suite_contains "wayformer_cityholdout"; then
      run_one "wayformer_cityholdout_${city}" "wayformer" "$(ckpt_candidate_for "WAYFORMER_CITYHOLDOUT_${city_env}" "${CKPT_ROOT}/sind_wayformer_cityholdout_${city}")" "sind_wayformer_cityholdout_${city}_eval" "false" "${BASELINE_SCRATCH_ROOT}" "false" "city_holdout" "${baseline_tag}" "${city}"
    fi
    if suite_contains "wayformer_signal_cityholdout"; then
      run_one "wayformer_signal_cityholdout_${city}" "wayformer" "$(ckpt_candidate_for "WAYFORMER_SIGNAL_CITYHOLDOUT_${city_env}" "${CKPT_ROOT}/sind_wayformer_signal_cityholdout_${city}")" "sind_wayformer_signal_cityholdout_${city}_eval" "true" "${SIGNAL_SCRATCH_ROOT}" "true" "city_holdout" "${signal_tag}" "${city}"
    fi
  done
  echo "[done] suite outputs are under ${OUTPUT_ROOT}"
else
  if [ -z "${CKPT_PATH}" ]; then
    echo "[error] CKPT_PATH is required unless RUN_SUITE=true" >&2
    exit 1
  fi
  LANE_CONTROL_MAP_TOKENS="${LANE_CONTROL_MAP_TOKENS:-${USE_LANE_CONTROL_STATE_IN_MAP_TOKENS:-${SIGNAL}}}"
  CITY_HOLDOUT_TAG="${CITY_HOLDOUT_TAG:-${SIGNAL_CITY_HOLDOUT_TAG}}"
  run_one "single" "${METHOD}" "${CKPT_PATH}" "${EXP_NAME}" "${USE_TRAFFIC_LIGHT_TOKENS}" "${SCRATCH_ROOT}" "${LANE_CONTROL_MAP_TOKENS}" "${SPLIT_MODE}" "${CITY_HOLDOUT_TAG}" ""
fi
