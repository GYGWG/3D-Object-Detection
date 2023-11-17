#!/bin/bash

# generate static model
python tools/export.py \
    --config configs/centerpoint/centerpoint_voxels_0075voxel_nuscenes_10sweep.yml \
    --model model/CenterPoint/model.pdparams \
    --save_dir model/CenterPoint/best_model
