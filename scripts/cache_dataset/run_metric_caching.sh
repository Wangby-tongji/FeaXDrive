TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/navdata/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/path/to/CVLA/navsim/exp/navlt_driverl"
export NAVSIM_DEVKIT_ROOT="/path/to/CVLA"
export OPENSCENE_DATA_ROOT="/path/to/navdata"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH