#!/usr/bin/env bash

# ============================================================
# FeaXDrive local environment for paper reproduction
# Local only. Do NOT commit this file to GitHub.
# ============================================================

# ------------------------------
# Repository root
# ------------------------------
export FEAXDRIVE_ROOT="/path/to/FeaXDrive"
export NAVSIM_DEVKIT_ROOT="${FEAXDRIVE_ROOT}"
export PYTHONPATH="${FEAXDRIVE_ROOT}:${PYTHONPATH:-}"

# ------------------------------
# NAVSIM / OpenScene dataset
# ------------------------------
# This directory contains:
#   navsim_logs/
#   sensor_blobs/
#   maps/
export OPENSCENE_DATA_ROOT="/path/to/navdata"

# ------------------------------
# nuPlan HD maps
# ------------------------------
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/navdata/maps/nuplan-maps-v1.0"

# ------------------------------
# Experiment outputs
# ------------------------------
# Keep new reproduction outputs under the clean FeaXDrive repo,
# instead of writing into the original CVLA directory.
export NAVSIM_EXP_ROOT="${FEAXDRIVE_ROOT}/navsim/exp"

# ------------------------------
# Cache paths
# ------------------------------
export FEAX_TRAIN_CACHE="/path/to/navcache/navtrain_hidden_state"
export FEAX_METRIC_CACHE_NAVTEST="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export FEAX_METRIC_CACHE_NAVTRAIN="${NAVSIM_EXP_ROOT}/metric_cache_navtrain"

# ------------------------------
# VLM checkpoint
# ------------------------------
# If this path does not exist, use the search command below to find it.
export VLM_CKPT="/path/to/ReCogDrive-VLM-2B"

# ------------------------------
# Planner checkpoints
# ------------------------------
# Use your previous 88.ckpt for first evaluation/debug.
export FEAX_IL_CKPT="/path/to/CVLA/navsim/exp/navtrain/training_recogdrive_agent/orin/lightning_logs/version_0/checkpoints/88.ckpt"

# Fill this after FA-GRPO training.
export FEAX_FA_GRPO_CKPT=""

# ------------------------------
# Distributed defaults
# ------------------------------
export MASTER_PORT="${MASTER_PORT:-63669}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

# ============================================================
# FeaXDrive paper reproduction checkpoints
# All ckpts are kept directly under FeaXDrive/navsim/exp
# ============================================================

export CKPT_TABLE4_BASELINE_EPS="/path/to/FeaXDrive/navsim/exp/navlt_eps/training_recogdrive_agent_eps/2026.03.16.01.08.18/lightning_logs/version_1285754/checkpoints/best_eps.ckpt"

export CKPT_FEAX_IL="/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/best_dyn.ckpt"

export CKPT_FEAX_GRPO="/path/to/FeaXDrive/navsim/exp/navlt_dynrl/training_recogdrive_agent_dynrl_4gpu/v1/lightning_logs/version_1322799/checkpoints/best_rl.ckpt"

export CKPT_FEAX_FA_GRPO="/path/to/FeaXDrive/navsim/exp/navlt_dynrl/training_recogdrive_agent_dynrl_4gpu/FeaRL/lightning_logs/version_1346196/checkpoints/0.ckpt"

export FEAX_IL_CKPT="${CKPT_FEAX_IL}"
export FEAX_FA_GRPO_CKPT="${CKPT_FEAX_FA_GRPO}"
