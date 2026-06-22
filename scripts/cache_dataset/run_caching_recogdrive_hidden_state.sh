TRAIN_TEST_SPLIT=navtrain #navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/navdata/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/path/to/CVLA/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/CVLA"
export OPENSCENE_DATA_ROOT="/path/to/navdata"
CACHE_PATH=$NAVSIM_EXP_ROOT/navcache/navtrain_hidden_state

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0


export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"

torchrun \
    --nproc_per_node=1 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/path/to/ReCogDrive-VLM-2B" \
    cache_path=$CACHE_PATH  > caching_dataset.txt 2>&1
    