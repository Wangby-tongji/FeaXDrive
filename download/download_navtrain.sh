#!/usr/bin/env bash
set -euo pipefail

wget -c https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz
mv openscene-v1.1/meta_datas trainval_navsim_logs
rm -r openscene-v1.1

mkdir -p trainval_sensor_blobs/trainval
for split in {1..4}; do
    aria2c -c -x 4 -s 4 -k 1M \
      -o "navtrain_current_${split}.tgz" \
        "https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz"
    echo "Extracting file navtrain_current_${split}.tgz"
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz

    rsync -rv current_split_${split}/* trainval_sensor_blobs/trainval
    rm -rf current_split_${split}
done

# The public NAVSIM release no longer requires the historical sensor archives.
