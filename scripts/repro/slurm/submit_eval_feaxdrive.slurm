#!/bin/bash
#SBATCH --job-name=eval_main
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/feax_env.sh"
cd "${FEAX_ROOT}"
mkdir -p "${NAVSIM_EXP_ROOT}/logs"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
export VLM_CKPT="${VLM_CKPT:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

# ------------------------------------------------------------------
# Unified FeaXDrive evaluation modes
#   eps/orin      : eps-pred baseline, no drivable guidance
#   dt            : x0-pred-only baseline, no drivable guidance
#   dyn           : dt + dynamics/IL checkpoint, no drivable guidance
#   drivedyn      : dyn checkpoint + data_map drivable guidance
#   fa            : FA-GRPO checkpoint + data_map drivable guidance
# Aliases: il -> dyn; il_guidance/il_g/guidance -> drivedyn; fa_guidance/fa_grpo/fagrpo -> fa
# ------------------------------------------------------------------
SPLIT="${SPLIT:-navtest}"
MODE="${MODE:-drivedyn}"
case "${MODE}" in
  orin) MODE="eps" ;;
  il) MODE="dyn" ;;
  il_guidance|il_g|guidance) MODE="drivedyn" ;;
  fa_guidance|fa_grpo|fagrpo) MODE="fa" ;;
esac

BUILD_METRIC_CACHE="${BUILD_METRIC_CACHE:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DRIVABLE_SDF_SOURCE="${DRIVABLE_SDF_SOURCE:-data_map}"
USE_DRIVABLE_GUIDANCE="${USE_DRIVABLE_GUIDANCE:-auto}"
DRIVABLE_SDF_MAP_RADIUS="${DRIVABLE_SDF_MAP_RADIUS:-100.0}"

if [ -z "${METRIC_CACHE:-}" ]; then
  if [ -d "${NAVSIM_EXP_ROOT}/metric_cache_${SPLIT}" ]; then
    METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache_${SPLIT}"
  elif [ -d "${NAVSIM_EXP_ROOT}/metric_cache" ]; then
    METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache"
  else
    METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache_${SPLIT}"
  fi
fi

CKPT_FEAX_DYN="${CKPT_FEAX_DYN:-${NAVSIM_EXP_ROOT}/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt}"
CKPT_FEAX_IL="${CKPT_FEAX_IL:-${CKPT_FEAX_DYN}}"
CKPT_FEAX_DT="${CKPT_FEAX_DT:-${NAVSIM_EXP_ROOT}/navlt_x0/training_recogdrive_agent_x0/2026.03.22.02.09.37/lightning_logs/version_1302349/checkpoints/best_x0_100.ckpt}"
CKPT_EPS="${CKPT_EPS:-${NAVSIM_EXP_ROOT}/navlt_eps/training_recogdrive_agent_eps/2026.03.16.01.08.18/lightning_logs/version_1285754/checkpoints/best_eps.ckpt}"
CKPT_FEAX_FA_GRPO="${CKPT_FEAX_FA_GRPO:-${CKPT_FEAX_FA:-${NAVSIM_EXP_ROOT}/navlt_dynrl/training_recogdrive_agent_dynrl_4gpu/FeaRL/lightning_logs/version_1346196/checkpoints/0.ckpt}}"

case "${MODE}" in
  eps)
    CKPT="${CKPT_EPS}"
    GRPO="False"
    DIFF_TYPE="eps"
    DEFAULT_USE_DRIVABLE_GUIDANCE="false"
    PLANNER_VARIANT="standard"
    EXP_NAME="eval_feaxdrive_eps_orin"
    ;;
  dt)
    CKPT="${CKPT_FEAX_DT}"
    GRPO="False"
    DIFF_TYPE="x0"
    DEFAULT_USE_DRIVABLE_GUIDANCE="false"
    PLANNER_VARIANT="standard"
    EXP_NAME="eval_feaxdrive_dt_x0_only"
    ;;
  dyn)
    CKPT="${CKPT_FEAX_DYN}"
    GRPO="False"
    DIFF_TYPE="x0"
    DEFAULT_USE_DRIVABLE_GUIDANCE="false"
    PLANNER_VARIANT="standard"
    EXP_NAME="eval_feaxdrive_dyn"
    ;;
  drivedyn)
    CKPT="${CKPT_FEAX_DYN}"
    GRPO="False"
    DIFF_TYPE="x0"
    DEFAULT_USE_DRIVABLE_GUIDANCE="true"
    PLANNER_VARIANT="standard"
    EXP_NAME="eval_feaxdrive_drivedyn"
    ;;
  fa)
    CKPT="${CKPT_FEAX_FA_GRPO}"
    GRPO="True"
    DIFF_TYPE="x0"
    DEFAULT_USE_DRIVABLE_GUIDANCE="true"
    PLANNER_VARIANT="standard"
    EXP_NAME="eval_feaxdrive_fa"
    ;;
  *)
    echo "ERROR: unknown MODE=${MODE}. Use eps | dt | dyn | drivedyn | fa"
    exit 1
    ;;
esac

if [ "${USE_DRIVABLE_GUIDANCE}" = "auto" ]; then
  USE_DRIVABLE_GUIDANCE="${DEFAULT_USE_DRIVABLE_GUIDANCE}"
fi

case "${DRIVABLE_SDF_SOURCE}" in
  data_map)
    ;;
  *)
    echo "ERROR: unsupported DRIVABLE_SDF_SOURCE=${DRIVABLE_SDF_SOURCE}"
    echo "This unified FeaXDrive release only supports DRIVABLE_SDF_SOURCE=data_map."
    exit 8
    ;;
esac

AGENT_CFG="feaxdrive_v4drive_agent"

echo "============================================================"
echo "EVAL MODE=${MODE}"
echo "SPLIT=${SPLIT}"
echo "CKPT=${CKPT}"
echo "GRPO=${GRPO}"
echo "DIFF_TYPE=${DIFF_TYPE}"
echo "PLANNER_VARIANT=${PLANNER_VARIANT}"
echo "DRIVABLE_SDF_SOURCE=${DRIVABLE_SDF_SOURCE}"
echo "USE_DRIVABLE_GUIDANCE=${USE_DRIVABLE_GUIDANCE}"
echo "METRIC_CACHE=${METRIC_CACHE}"
echo "VLM_CKPT=${VLM_CKPT}"
echo "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "EXP_NAME=${EXP_NAME}"
echo "============================================================"

ls -lh "${CKPT}"
feax_require_dir OPENSCENE_DATA_ROOT
feax_require_dir NUPLAN_MAPS_ROOT

if [ ! -d "${METRIC_CACHE}" ] || [ "$(find "${METRIC_CACHE}" -type f 2>/dev/null | wc -l)" -lt 10 ]; then
  echo "Metric cache not found or too small: ${METRIC_CACHE}"
  if [ "${BUILD_METRIC_CACHE}" = "1" ]; then
    echo "Building metric cache first..."
    python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
      train_test_split="${SPLIT}" \
      cache.cache_path="${METRIC_CACHE}"
  else
    echo "Set BUILD_METRIC_CACHE=1 to build it automatically."
    exit 2
  fi
fi

echo "==== Safety check: final evaluation must use standard PDM comfort ===="
test -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_feaxdrive.py"
test -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py"
grep -n "pdm_comfort_metrics" \
  navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
if grep -q "pdm_comfort_metrics_rl\|feaxdrive_fagrpo_comfort_metrics" navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py; then
  echo "ERROR: final evaluation scorer was polluted by RL/FA-GRPO comfort metrics"
  exit 3
fi

echo "==== Python syntax check for active eval files ===="
python -m py_compile \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_feaxdrive.py" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/agents/feaxdrive/feaxdrive_agent_entry_v4drive.py" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/agents/feaxdrive/feaxdrive_diffusion_planner.py" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_feaxdrive.py" \
  train_test_split="${SPLIT}" \
  metric_cache_path="${METRIC_CACHE}" \
  agent="${AGENT_CFG}" \
  agent.checkpoint_path="${CKPT}" \
  agent.vlm_path="${VLM_CKPT}" \
  agent.cam_type=single \
  agent.grpo="${GRPO}" \
  agent.metric_cache_path="${METRIC_CACHE}" \
  agent.reference_policy_checkpoint="${CKPT_FEAX_DYN}" \
  agent.cache_hidden_state=False \
  agent.vlm_type=internvl \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.diffusion_pred_type="${DIFF_TYPE}" \
  agent.planner_variant="${PLANNER_VARIANT}" \
  agent.drivable_sdf_source="${DRIVABLE_SDF_SOURCE}" \
  agent.use_drivable_guidance="${USE_DRIVABLE_GUIDANCE}" \
  agent.drivable_sdf_map_radius="${DRIVABLE_SDF_MAP_RADIUS}" \
  agent.drivable_guidance_last_k=3 \
  agent.drivable_guidance_step_size=0.05 \
  agent.drivable_guidance_margin_m=0.50 \
  agent.drivable_guidance_only_offroad=true \
  agent.drivable_guidance_offroad_eps_m=-0.40 \
  agent.drivable_guidance_beta_m=0.30 \
  agent.drivable_sdf_resolution=0.20 \
  agent.drivable_sdf_x_min=-10.0 \
  agent.drivable_sdf_x_max=80.0 \
  agent.drivable_sdf_y_min=-30.0 \
  agent.drivable_sdf_y_max=30.0 \
  agent.drivable_guidance_heading_scale=0.5 \
  dynamics_eval.enable=true \
  dynamics_eval.dt=0.5 \
  dynamics_eval.kappa_max=0.166 \
  dynamics_eval.kappa_geo_max=0.166 \
  dynamics_eval.kappa_adapt=true \
  dynamics_eval.a_lat_max=6.0 \
  dynamics_eval.smooth_ksize=5 \
  experiment_name="${EXP_NAME}"
