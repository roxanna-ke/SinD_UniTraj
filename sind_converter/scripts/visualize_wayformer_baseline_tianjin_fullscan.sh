#!/bin/bash
#SBATCH --job-name=wf_base_tianjin_vis
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

# Run only Wayformer baseline record-level prediction visualization for Tianjin,
# scanning the full validation/test cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"

export PROJECT_ROOT
export RUN_SUITE=true
export SUITE_INCLUDE=wayformer_baseline
export CITY_HOLDOUT_NAMES=Tianjin

export MAX_VAL_DATA_NUM=null
export NUM_IMAGES="${NUM_IMAGES:-200000}"

export AGGREGATE_VISUALIZATION=true
export AGGREGATE_ONLY=true
export AGGREGATE_MIN_TRACKS="${AGGREGATE_MIN_TRACKS:-3}"
export AGGREGATE_MAX_TRACKS="${AGGREGATE_MAX_TRACKS:-8}"
export AGGREGATE_MIN_TRACK_DISTANCE="${AGGREGATE_MIN_TRACK_DISTANCE:-4.0}"
export AGGREGATE_MIN_DISPLACEMENT="${AGGREGATE_MIN_DISPLACEMENT:-2.0}"
export AGGREGATE_MIN_TOTAL_STEPS="${AGGREGATE_MIN_TOTAL_STEPS:-81}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/prediction_visualizations}"

echo "[info] dedicated run: Wayformer baseline, record-level, Tianjin only"
echo "[info] full scan: MAX_VAL_DATA_NUM=${MAX_VAL_DATA_NUM}, NUM_IMAGES=${NUM_IMAGES}"
echo "[info] output: ${OUTPUT_ROOT}/sind_wayformer_baseline_vis"

bash "${SCRIPT_DIR}/prediction_visualize_sind.sh"
