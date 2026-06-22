#!/bin/bash
set -euo pipefail

# ===== conda =====
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate navsim-wby

# ===== paths =====
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/navdata/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/path/to/CVLA/navsim/exp/navcache"
export NAVSIM_DEVKIT_ROOT="/path/to/CVLA"
export OPENSCENE_DATA_ROOT="/path/to/navdata"

VLM_CKPT="${VLM_CKPT:-/path/to/ReCogDrive-VLM-2B}"
CACHE_PATH="${CACHE_PATH:-/path/to/navcache/navtrain_hidden_state}"
MASTER_PORT="${MASTER_PORT:-29511}"

# ===== runtime env =====
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

mkdir -p "$CACHE_PATH"

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "VLM_CKPT: ${VLM_CKPT}"
echo "CACHE_PATH: ${CACHE_PATH}"

cd "$NAVSIM_DEVKIT_ROOT"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py" \
  agent=recogdrive_agent \
  experiment_name=recogdrive_agent_cache \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.cache_mode=True \
  agent.vlm_path="$VLM_CKPT" \
  agent.vlm_type=internvl \
  agent.vlm_size=small \
  train_test_split="$TRAIN_TEST_SPLIT" \
  cache_path="$CACHE_PATH"