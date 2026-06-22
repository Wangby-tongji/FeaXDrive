#!/bin/bash
#SBATCH --job-name=cache_main
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/FeaXDrive/navsim/exp/logs/%x-%j.out
#SBATCH --error=/path/to/FeaXDrive/navsim/exp/logs/%x-%j.err

set -eo pipefail

# ======================================================================
# FeaXDrive unified cache scheduler
#
# CACHE_MODE:
#   check        : only print cache status
#   hidden_train : build/fill VLM hidden-state cache using source-style run_dataset_caching_multi_node.py
#   metric_train : build PDM metric cache for FA-GRPO training reward
#   metric_eval  : build PDM metric cache for final eval split, default navtest
#   all_train    : hidden_train + metric_train
#   all_eval     : metric_eval
#   all          : hidden_train + metric_train + metric_eval
#
# Cache semantics:
# - hidden_train is needed by eps/dt/dyn/FA-GRPO training.
# - hidden_train follows the original CVLA/ReCogDrive cache chain:
#     run_dataset_caching_multi_node.py -> SceneLoader(load_image_path=True) -> Dataset.cache_dataset().
# - metric_train is needed by FA-GRPO training reward.
# - metric_eval is needed by final PDM evaluation.
# - data_map drivable guidance does not require a persistent SDF cache.
# ======================================================================

cd /path/to/FeaXDrive
mkdir -p navsim/exp/logs

set +u
source ~/.bashrc
conda activate navsim-wby
set -u

if [ -f scripts/envs/env_local.sh ]; then
  source scripts/envs/env_local.sh
fi

export FEAX_ROOT="${FEAX_ROOT:-/path/to/FeaXDrive}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${FEAX_ROOT}}"
export NAVSIM_EXP_BASE="${NAVSIM_EXP_BASE:-${FEAX_ROOT}/navsim/exp}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${NAVSIM_EXP_BASE}}"
export PYTHONPATH="${FEAX_ROOT}:${PYTHONPATH:-}"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/navdata}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/navdata/maps/nuplan-maps-v1.0}"
export VLM_CKPT="${VLM_CKPT:-/path/to/ReCogDrive-VLM-2B}"
# Agent used only for hidden-state cache generation.
HIDDEN_AGENT="${HIDDEN_AGENT:-feaxdrive_v4drive_agent}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CACHE_MODE="${CACHE_MODE:-check}"
case "${CACHE_MODE}" in
  hidden|hidden_state|hidden-cache) CACHE_MODE="hidden_train" ;;
  metric|metric-cache) CACHE_MODE="metric_eval" ;;
  metric_fa|metric_rl|metric_driverl) CACHE_MODE="metric_train" ;;
esac

# Splits.
HIDDEN_SPLIT="${HIDDEN_SPLIT:-navtrain}"
TRAIN_METRIC_SPLIT="${TRAIN_METRIC_SPLIT:-navtrain}"
EVAL_SPLIT="${SPLIT:-${EVAL_SPLIT:-navtest}}"

# Paths.
HIDDEN_CACHE="${HIDDEN_CACHE:-/path/to/navcache/navtrain_hidden_state}"
METRIC_CACHE_TRAIN="${METRIC_CACHE_TRAIN:-${NAVSIM_EXP_BASE}/navlt_driverl/metric_cache}"
METRIC_CACHE_EVAL="${METRIC_CACHE_EVAL:-${NAVSIM_EXP_BASE}/metric_cache_${EVAL_SPLIT}}"

# Hidden cache generation defaults.
# FORCE_HIDDEN_CACHE=false means fill missing tokens only. Set true to recompute all.
FORCE_HIDDEN_CACHE="${FORCE_HIDDEN_CACHE:-false}"
# Avoid expensive du/find over large /ssdfs hidden-state cache during status checks.
# Set CHECK_HIDDEN_CACHE=true only when you explicitly want to inspect it.
CHECK_HIDDEN_CACHE="${CHECK_HIDDEN_CACHE:-false}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-1}"
HIDDEN_NUM_WORKERS="${HIDDEN_NUM_WORKERS:-4}"
HIDDEN_NPROC_PER_NODE="${HIDDEN_NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29649}"
SEED="${SEED:-0}"

echo "============================================================"
echo "CACHE_MODE=${CACHE_MODE}"
echo "FEAX_ROOT=${FEAX_ROOT}"
echo "OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "VLM_CKPT=${VLM_CKPT}"
echo "HIDDEN_AGENT=${HIDDEN_AGENT}"
echo "HIDDEN_SPLIT=${HIDDEN_SPLIT}"
echo "HIDDEN_CACHE=${HIDDEN_CACHE}"
echo "FORCE_HIDDEN_CACHE=${FORCE_HIDDEN_CACHE}"
echo "TRAIN_METRIC_SPLIT=${TRAIN_METRIC_SPLIT}"
echo "METRIC_CACHE_TRAIN=${METRIC_CACHE_TRAIN}"
echo "EVAL_SPLIT=${EVAL_SPLIT}"
echo "METRIC_CACHE_EVAL=${METRIC_CACHE_EVAL}"
echo "============================================================"

test -d "${FEAX_ROOT}"
test -d "${OPENSCENE_DATA_ROOT}"
test -d "${NUPLAN_MAPS_ROOT}"
test -d "${VLM_CKPT}"

cache_status() {
  echo "==== Cache status ===="
  echo "[SKIP] hidden cache deep status: ${HIDDEN_CACHE}"
  if [ "${CHECK_HIDDEN_CACHE}" = "true" ]; then
    if [ -d "${HIDDEN_CACHE}" ]; then
      echo "[OK] ${HIDDEN_CACHE}"
      du -sh "${HIDDEN_CACHE}" || true
      echo "files=$(find "${HIDDEN_CACHE}" -type f 2>/dev/null | wc -l)"
    else
      echo "[MISSING] ${HIDDEN_CACHE}"
    fi
    echo "----"
  fi

  for p in "${METRIC_CACHE_TRAIN}" "${METRIC_CACHE_EVAL}"; do
    if [ -d "$p" ]; then
      echo "[OK] $p"
      du -sh "$p" || true
      echo "files=$(find "$p" -type f 2>/dev/null | wc -l)"
    else
      echo "[MISSING] $p"
    fi
    echo "----"
  done
}

build_hidden_train_cache() {
  echo "==== Build/fill hidden-state training cache: source-style ===="
  mkdir -p "${HIDDEN_CACHE}"

  # IMPORTANT:
  # Use the original CVLA/ReCogDrive hidden-cache entry, not run_training_feaxdrive_rl.py.
  # run_dataset_caching_multi_node.py constructs SceneLoader(..., load_image_path=True),
  # so cameras[-1].cam_f0.image remains an image path for InternVL preprocessing.
  # This avoids the ndarray-as-filename error:
  #   OSError: [Errno 36] File name too long: '[[[156 199 231] ...]]'

  test -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py"

  echo "==== Python syntax check for source-style hidden cache chain ===="
  python -m py_compile \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/training/dataset.py" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/agents/feaxdrive/feaxdrive_features.py" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/agents/feaxdrive/utils/internvl_preprocess.py"

  echo "==== Source-chain safety check ===="
  grep -n "load_image_path=True" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py"

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${HIDDEN_NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
    agent="${HIDDEN_AGENT}" \
    experiment_name="cache_hidden_${HIDDEN_SPLIT}" \
    seed="${SEED}" \
    agent.vlm_path="${VLM_CKPT}" \
    agent.cam_type=single \
    agent.vlm_type=internvl \
    agent.dit_type=small \
    agent.vlm_size=small \
    agent.sampling_method=ddim \
    agent.diffusion_pred_type=x0 \
    agent.planner_variant=standard \
    agent.grpo=false \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.use_drivable_guidance=false \
    train_test_split="${HIDDEN_SPLIT}" \
    cache_path="${HIDDEN_CACHE}" \
    force_cache_computation="${FORCE_HIDDEN_CACHE}"

  echo "==== Hidden cache finished ===="
  echo "HIDDEN_CACHE=${HIDDEN_CACHE}"
  echo "internvl_feature.gz files:"
  find "${HIDDEN_CACHE}" -name "internvl_feature.gz" 2>/dev/null | wc -l
  echo "trajectory_target.gz files:"
  find "${HIDDEN_CACHE}" -name "trajectory_target.gz" 2>/dev/null | wc -l
  echo "first files:"
  find "${HIDDEN_CACHE}" -type f 2>/dev/null | head -20
}

build_metric_cache() {
  local split="$1"
  local out="$2"
  echo "==== Build PDM metric cache ===="
  echo "split=${split}"
  echo "cache=${out}"
  mkdir -p "${out}"

  if [ ! -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" ]; then
    echo "ERROR: ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py not found."
    echo "Your eval slurm calls this script when BUILD_METRIC_CACHE=1; please check the project tree."
    exit 11
  fi

  python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split="${split}" \
    cache.cache_path="${out}"

  echo "==== Metric cache finished ===="
  du -sh "${out}" || true
  find "${out}" -type f | wc -l
}

case "${CACHE_MODE}" in
  check)
    cache_status
    ;;
  hidden_train)
    build_hidden_train_cache
    cache_status
    ;;
  metric_train)
    build_metric_cache "${TRAIN_METRIC_SPLIT}" "${METRIC_CACHE_TRAIN}"
    cache_status
    ;;
  metric_eval)
    build_metric_cache "${EVAL_SPLIT}" "${METRIC_CACHE_EVAL}"
    cache_status
    ;;
  all_train)
    build_hidden_train_cache
    build_metric_cache "${TRAIN_METRIC_SPLIT}" "${METRIC_CACHE_TRAIN}"
    cache_status
    ;;
  all_eval)
    build_metric_cache "${EVAL_SPLIT}" "${METRIC_CACHE_EVAL}"
    cache_status
    ;;
  all)
    build_hidden_train_cache
    build_metric_cache "${TRAIN_METRIC_SPLIT}" "${METRIC_CACHE_TRAIN}"
    build_metric_cache "${EVAL_SPLIT}" "${METRIC_CACHE_EVAL}"
    cache_status
    ;;
  *)
    echo "ERROR: unknown CACHE_MODE=${CACHE_MODE}"
    echo "Use: check | hidden_train | metric_train | metric_eval | all_train | all_eval | all"
    exit 2
    ;;
esac

echo "OK: cache scheduler finished."
