#!/bin/bash
#SBATCH --job-name=train_main
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/feax_env.sh"
cd "${FEAX_ROOT}"
mkdir -p "${NAVSIM_EXP_ROOT}/logs"

# ======================================================================
# FeaXDrive paper training scheduler
#
# MODE=eps : epsilon-pred baseline
# MODE=dt  : x0-pred / trajectory-centric baseline
# MODE=dyn : x0-pred + feasibility training, no inference guidance
# MODE=fa  : FA-GRPO based on a trained dyn checkpoint
#
# Important:
# - There is no MODE=drivedyn training. drivedyn is eval-time:
#     dyn checkpoint + data_map drivable guidance.
# - use_constraint_projection is deprecated in the current official chain.
#   This script does not pass agent.use_constraint_projection.
# - eps/dt/dyn are trained from scratch by default. Pass INIT_CKPT only if
#   you intentionally want to resume/finetune.
# ======================================================================

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
export VLM_CKPT="${VLM_CKPT:-}"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"

MODE="${MODE:-${TRAIN_MODE:-dyn}}"
case "${MODE}" in
  orin) MODE="eps" ;;
  x0) MODE="dt" ;;
  il) MODE="dyn" ;;
  fa_grpo|fagrpo|driverl) MODE="fa" ;;
esac

SPLIT="${SPLIT:-navtrain}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29633}"
USER_LR="${LR:-}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
HIDDEN_CACHE="${HIDDEN_CACHE:-${NAVSIM_EXP_BASE}/cache_navtrain_hidden_state}"
USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-true}"

# Paper IL-stage training defaults.
IL_MAX_EPOCHS="${IL_MAX_EPOCHS:-100}"
IL_BATCH_SIZE="${IL_BATCH_SIZE:-32}"

# FA-GRPO defaults. Override if reproducing a longer RL run.
FA_MAX_EPOCHS="${FA_MAX_EPOCHS:-2}"
FA_BATCH_SIZE="${FA_BATCH_SIZE:-4}"

# FA-GRPO reward defaults from the source chain.
FA_REWARD_PROGRESS_WEIGHT="${FA_REWARD_PROGRESS_WEIGHT:-10.0}"
FA_REWARD_TTC_WEIGHT="${FA_REWARD_TTC_WEIGHT:-5.0}"
FA_REWARD_COMFORTABLE_WEIGHT="${FA_REWARD_COMFORTABLE_WEIGHT:-10.0}"
FA_GAMMA_DENOISING="${FA_GAMMA_DENOISING:-0.6}"
FA_BC_COEFF="${FA_BC_COEFF:-0.1}"

DRIVABLE_SDF_SOURCE="${DRIVABLE_SDF_SOURCE:-data_map}"
DRIVABLE_SDF_MAP_RADIUS="${DRIVABLE_SDF_MAP_RADIUS:-100.0}"

# Proven paper dyn checkpoint for FA-GRPO initialization/reference by default.
CKPT_DYN_PAPER="${CKPT_DYN_PAPER:-${NAVSIM_EXP_BASE}/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt}"
CKPT_FEAX_IL="${CKPT_FEAX_IL:-${CKPT_DYN_PAPER}}"

# Metric cache is needed by FA-GRPO.
if [ -z "${METRIC_CACHE:-}" ]; then
  if [ -d "${NAVSIM_EXP_BASE}/navlt_driverl/metric_cache" ]; then
    METRIC_CACHE="${NAVSIM_EXP_BASE}/navlt_driverl/metric_cache"
  elif [ -d "${NAVSIM_EXP_BASE}/metric_cache_${SPLIT}" ]; then
    METRIC_CACHE="${NAVSIM_EXP_BASE}/metric_cache_${SPLIT}"
  elif [ -d "${NAVSIM_EXP_BASE}/metric_cache" ]; then
    METRIC_CACHE="${NAVSIM_EXP_BASE}/metric_cache"
  else
    METRIC_CACHE="${NAVSIM_EXP_BASE}/metric_cache_${SPLIT}"
  fi
fi

case "${MODE}" in
  eps)
    EXP_GROUP="${EXP_GROUP:-navlt_eps}"
    EXP_NAME="${EXP_NAME:-training_recogdrive_agent_eps}"
    AGENT_NAME="${AGENT_NAME:-feaxdrive_v4drive_agent}"
    PLANNER_VARIANT="standard"
    DIFF_TYPE="eps"
    TRAIN_GRPO="false"
    USE_DRIVABLE_GUIDANCE="false"
    MAX_EPOCHS="${MAX_EPOCHS:-${IL_MAX_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-${IL_BATCH_SIZE}}"
    INIT_CKPT="${INIT_CKPT:-}"
    ;;
  dt)
    EXP_GROUP="${EXP_GROUP:-navlt_x0}"
    EXP_NAME="${EXP_NAME:-training_recogdrive_agent_x0}"
    AGENT_NAME="${AGENT_NAME:-feaxdrive_v4drive_agent}"
    PLANNER_VARIANT="standard"
    DIFF_TYPE="x0"
    TRAIN_GRPO="false"
    USE_DRIVABLE_GUIDANCE="false"
    MAX_EPOCHS="${MAX_EPOCHS:-${IL_MAX_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-${IL_BATCH_SIZE}}"
    INIT_CKPT="${INIT_CKPT:-}"
    ;;
  dyn)
    EXP_GROUP="${EXP_GROUP:-navlt_dyn}"
    EXP_NAME="${EXP_NAME:-training_recogdrive_agent_dyn}"
    AGENT_NAME="${AGENT_NAME:-feaxdrive_v4drive_agent}"
    PLANNER_VARIANT="standard"
    DIFF_TYPE="x0"
    TRAIN_GRPO="false"
    USE_DRIVABLE_GUIDANCE="false"
    MAX_EPOCHS="${MAX_EPOCHS:-${IL_MAX_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-${IL_BATCH_SIZE}}"
    INIT_CKPT="${INIT_CKPT:-}"
    ;;
  fa)
    EXP_GROUP="${EXP_GROUP:-navlt_fa_grpo}"
    EXP_NAME="${EXP_NAME:-training_feaxdrive_fa_grpo}"
    AGENT_NAME="${AGENT_NAME:-feaxdrive_v4drive_fagrpo_agent}"
    PLANNER_VARIANT="fagrpo"
    DIFF_TYPE="x0"
    TRAIN_GRPO="true"
    USE_DRIVABLE_GUIDANCE="true"
    MAX_EPOCHS="${MAX_EPOCHS:-${FA_MAX_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-${FA_BATCH_SIZE}}"
    INIT_CKPT="${INIT_CKPT:-${CKPT_FEAX_IL}}"
    # Source FA-GRPO uses lr=1e-5 unless explicitly overridden.
    if [ -z "${USER_LR}" ]; then
      LR=1e-5
    fi
    ;;
  *)
    echo "ERROR: unknown MODE=${MODE}. Use eps | dt | dyn | fa"
    exit 1
    ;;
esac

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_BASE}/${EXP_GROUP}"
mkdir -p "${NAVSIM_EXP_ROOT}"

case "${DRIVABLE_SDF_SOURCE}" in
  data_map) ;;
  *)
    echo "ERROR: unsupported DRIVABLE_SDF_SOURCE=${DRIVABLE_SDF_SOURCE}; official chain uses data_map."
    exit 8
    ;;
esac

# Mode-aware Python training log filename. This only affects the log
# file name under the Hydra run directory; it does not change the
# training entry script or model behavior.
export FEAX_TRAIN_LOG_NAME="${FEAX_TRAIN_LOG_NAME:-run_training_feaxdrive_${MODE}.log}"

echo "============================================================"
echo "TRAIN MODE=${MODE}"
echo "EXP_GROUP=${EXP_GROUP}"
echo "EXP_NAME=${EXP_NAME}"
echo "NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "AGENT_NAME=${AGENT_NAME}"
echo "SPLIT=${SPLIT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "INIT_CKPT=${INIT_CKPT:-<none>}"
echo "HIDDEN_CACHE=${HIDDEN_CACHE}"
echo "METRIC_CACHE=${METRIC_CACHE}"
echo "VLM_CKPT=${VLM_CKPT}"
echo "DRIVABLE_SDF_SOURCE=${DRIVABLE_SDF_SOURCE}"
echo "USE_DRIVABLE_GUIDANCE=${USE_DRIVABLE_GUIDANCE}"
echo "DIFF_TYPE=${DIFF_TYPE}"
echo "PLANNER_VARIANT=${PLANNER_VARIANT}"
echo "TRAIN_GRPO=${TRAIN_GRPO}"
if [ "${MODE}" = "fa" ]; then
  echo "FA_REWARD_PROGRESS_WEIGHT=${FA_REWARD_PROGRESS_WEIGHT}"
  echo "FA_REWARD_TTC_WEIGHT=${FA_REWARD_TTC_WEIGHT}"
  echo "FA_REWARD_COMFORTABLE_WEIGHT=${FA_REWARD_COMFORTABLE_WEIGHT}"
  echo "FA_GAMMA_DENOISING=${FA_GAMMA_DENOISING}"
  echo "FA_BC_COEFF=${FA_BC_COEFF}"
fi
echo "============================================================"

feax_require_dir OPENSCENE_DATA_ROOT
feax_require_dir NUPLAN_MAPS_ROOT
test -d "${HIDDEN_CACHE}"

if [ -n "${INIT_CKPT}" ]; then
  ls -lh "${INIT_CKPT}"
fi

if [ "${MODE}" = "fa" ]; then
  test -d "${METRIC_CACHE}"
  ls -lh "${CKPT_FEAX_IL}"

  echo "==== FA-GRPO safety check ===="
  echo "-- FA-GRPO scorer should use isolated comfort metrics:"
  grep -n "feaxdrive_fagrpo_comfort_metrics" \
    navsim/planning/simulation/planner/pdm_planner/scoring/feaxdrive_fagrpo_scorer.py

  echo "-- FA-GRPO planner should use isolated score wrapper and source reward weights:"
  grep -n "pdm_score_feaxdrive_fagrpo\|feaxdrive_fagrpo_scorer\|progress_weight=10.0\|ttc_weight=5.0\|comfortable_weight=10.0\|bc_coeff" \
    navsim/agents/feaxdrive/feaxdrive_diffusion_planner_fagrpo.py | head -80

  echo "-- Standard eval scorer should remain standard:"
  grep -n "pdm_comfort_metrics" \
    navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py

  if grep -q "pdm_comfort_metrics_rl\|feaxdrive_fagrpo_comfort_metrics" \
    navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py; then
    echo "ERROR: standard final eval scorer was polluted by FA-GRPO comfort metrics"
    exit 3
  fi
fi

echo "==== Python syntax check for active training files ===="
python -m py_compile \
  navsim/planning/script/run_training_feaxdrive_rl.py \
  navsim/agents/feaxdrive/feaxdrive_agent_entry_v4drive.py \
  navsim/agents/feaxdrive/feaxdrive_agent_entry_v4drive_fagrpo.py \
  navsim/agents/feaxdrive/feaxdrive_diffusion_planner.py \
  navsim/agents/feaxdrive/trajectory_projector.py \
  navsim/evaluate/pdm_score_fagrpo.py \
  navsim/evaluate/pdm_score_feaxdrive_fagrpo.py \
  navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_fagrpo.py \
  navsim/planning/simulation/planner/pdm_planner/scoring/feaxdrive_fagrpo_scorer.py \
  navsim/planning/simulation/planner/pdm_planner/scoring/feaxdrive_fagrpo_comfort_metrics.py \
  navsim/agents/feaxdrive/feaxdrive_diffusion_planner_fagrpo.py

HYDRA_ARGS=(
  train_test_split="${SPLIT}"
  experiment_name="${EXP_NAME}"
  seed="${SEED}"

  agent="${AGENT_NAME}"
  agent.vlm_path="${VLM_CKPT}"
  agent.cam_type=single
  agent.vlm_type=internvl
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  agent.diffusion_pred_type="${DIFF_TYPE}"
  agent.planner_variant="${PLANNER_VARIANT}"
  agent.drivable_sdf_source="${DRIVABLE_SDF_SOURCE}"
  agent.use_drivable_guidance="${USE_DRIVABLE_GUIDANCE}"
  agent.drivable_sdf_map_radius="${DRIVABLE_SDF_MAP_RADIUS}"
  agent.grpo="${TRAIN_GRPO}"
  agent.lr="${LR}"
  agent.cache_hidden_state=true

  cache_path="${HIDDEN_CACHE}"
  use_cache_without_dataset="${USE_CACHE_WITHOUT_DATASET}"
  force_cache_computation=false

  dataloader.params.batch_size="${BATCH_SIZE}"
  dataloader.params.num_workers="${NUM_WORKERS}"
  dataloader.params.pin_memory=true
  dataloader.params.persistent_workers=true
  dataloader.params.prefetch_factor=2

  trainer.params.devices="${NPROC_PER_NODE}"
  trainer.params.num_nodes=1
  trainer.params.max_epochs="${MAX_EPOCHS}"
  trainer.params.check_val_every_n_epoch=1
  trainer.params.val_check_interval=1.0
  trainer.params.limit_train_batches=1.0
  trainer.params.limit_val_batches=1.0
  trainer.params.precision=bf16-mixed
  trainer.params.strategy=ddp_find_unused_parameters_false
  trainer.params.gradient_clip_val=1.0
  trainer.params.gradient_clip_algorithm=norm
  trainer.params.accumulate_grad_batches=1
  trainer.params.num_sanity_val_steps=0
)

# Add/resume checkpoint only when explicitly requested, except FA where it is required.
if [ -n "${INIT_CKPT}" ]; then
  HYDRA_ARGS+=(agent.checkpoint_path="${INIT_CKPT}")
fi

# FA-GRPO needs reference policy and metric cache.
if [ "${MODE}" = "fa" ]; then
  HYDRA_ARGS+=(
    agent.reference_policy_checkpoint="${CKPT_FEAX_IL}"
    agent.metric_cache_path="${METRIC_CACHE}"
    ++agent.grpo_progress_weight="${FA_REWARD_PROGRESS_WEIGHT}"
    ++agent.grpo_ttc_weight="${FA_REWARD_TTC_WEIGHT}"
    ++agent.grpo_comfortable_weight="${FA_REWARD_COMFORTABLE_WEIGHT}"
    ++agent.grpo_gamma_denoising="${FA_GAMMA_DENOISING}"
    ++agent.grpo_bc_coeff="${FA_BC_COEFF}"
  )
fi

# Mode-specific feasibility-training overrides.
# Use ++ so the override works whether the current YAML already contains the key or not.
# Do not pass use_constraint_projection: it is deprecated in the current official chain.
if [ "${MODE}" = "dyn" ]; then
  HYDRA_ARGS+=(
    ++agent.proj_dt=0.5
    ++agent.proj_kappa_max=0.166
    ++agent.proj_kappa_geo_max=0.166
    ++agent.proj_use_kappa_adapt=true
    ++agent.proj_a_lat_max=6.0
    ++agent.proj_smooth_ksize=5
    ++agent.proj_step_size=0.02
    ++agent.proj_lambda_kappa=1.0
    ++agent.proj_lambda_a_lat=1.0
    ++agent.proj_iters=1
    ++agent.proj_keep_start=true
    ++agent.lambda_dyn=0.01
    ++agent.lambda_proj=0.0
    ++agent.lambda_proj_supervise=0.0
    ++agent.proj_stop_grad=true
    ++agent.record_delta_pi_curve=false
  )
else
  HYDRA_ARGS+=(
    ++agent.lambda_dyn=0.0
    ++agent.lambda_proj=0.0
    ++agent.lambda_proj_supervise=0.0
  )
fi

echo "==== Hydra overrides ===="
printf '  %q\n' "${HYDRA_ARGS[@]}"

echo "==== Launch training ===="
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_feaxdrive_rl.py" \
  "${HYDRA_ARGS[@]}"
