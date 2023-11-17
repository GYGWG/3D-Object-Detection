#!/bin/bash

# For Car
python deploy/pointpillars/python/infer.py \
    --det_class Car \
    --model_file outputs/pointpillars_zw_car_1102/best_model/pointpillars.pdmodel \
    --params_file outputs/pointpillars_zw_car5/best_model/pointpillars.pdiparams \
    --lidar_file datasets/zw_anno_0914/KITTI_format/training/velodyne/load2-low-speed_sync_012100.bin \
    --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
    --voxel_size .16 .16 4 \
    --max_points_in_voxel 32 \
    --max_voxel_num 40000

# For Person
#python deploy/pointpillars/python/infer.py \
#    --det_class Person \
#    --model_file model/Person/output/pointpillars.pdmodel \
#    --params_file model/Person/output/pointpillars.pdiparams \
#    --lidar_file datasets/zw_0830/load1-low-speed_sync/lidar_bin/$1.bin \
#    --point_cloud_range 0 -19.84 -2.5 47.36 19.84 0.5 \
#    --voxel_size .16 .16 3 \
#    --max_points_in_voxel 100 \
#    --max_voxel_num 40000
