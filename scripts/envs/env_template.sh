#!/usr/bin/env bash
# Copy this file to scripts/envs/env_local.sh and modify the paths.
# The local copy is intentionally ignored by Git.

# Conda environment created by environment.yml. Override if needed.
export CONDA_ENV="${CONDA_ENV:-feaxdrive}"

# Repository root
export FEAX_ROOT="/path/to/FeaXDrive"
export NAVSIM_DEVKIT_ROOT="${FEAX_ROOT}"
export PYTHONPATH="${FEAX_ROOT}:${PYTHONPATH:-}"

# NAVSIM / OpenScene data root. Expected structure:
# ${OPENSCENE_DATA_ROOT}/navsim_logs/{trainval,test}
# ${OPENSCENE_DATA_ROOT}/sensor_blobs/{trainval,test}
export OPENSCENE_DATA_ROOT="/path/to/navsim_dataset"

# Experiment outputs and caches. Do not commit this folder to GitHub.
export NAVSIM_EXP_ROOT="/path/to/feaxdrive_exp"

# nuPlan maps
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/maps/nuplan-maps-v1.0"

# VLM and planner checkpoints
export VLM_CKPT="/path/to/ReCogDrive-VLM-2B"
export FEAX_IL_CKPT="/path/to/feaxdrive_il.ckpt"
export FEAX_FA_GRPO_CKPT="/path/to/feaxdrive_fa_grpo.ckpt"

# Cached training features / metric caches
export FEAX_TRAIN_CACHE="${NAVSIM_EXP_ROOT}/cache_navtrain_hidden_state"
export FEAX_METRIC_CACHE_NAVTEST="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export FEAX_METRIC_CACHE_NAVTRAIN="${NAVSIM_EXP_ROOT}/metric_cache_navtrain"

# Distributed defaults
export MASTER_PORT="${MASTER_PORT:-63669}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
