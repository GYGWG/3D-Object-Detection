python deploy/centerpoint/python/infer.py \
    --model_file model/CenterPoint/best_model/centerpoint.pdmodel \
    --params_file model/CenterPoint/best_model/centerpoint.pdiparams \
    --lidar_file datasets/nuScenes/lidar/n008-2018-09-18-12-53-31-0400__LIDAR_TOP__1537290211099302.pcd.bin \
    --num_point_dim 4