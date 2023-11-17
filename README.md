# 强对抗人、车检测

## <h2 id="1">简介</h2>
本项目采用PointPillars进行3D检测。PointPillars是目前工业界应用广泛的点云检测模型，其最主要的特点是检测速度和精度的平衡。PointPillars 在 [VoxelNet](https://arxiv.org/abs/1711.06396) 和 [SECOND](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)
 的基础上针对性能进行了优化，将点云转化为柱体（Pillars）表示，从而使得编码后的点云特征可以使用2D卷积神经网络进行检测任务。

## <h2 id="2">使用教程</h2>

### <h3 id="21">数据准备</h3>

- 目前Paddle3D中提供的PointPillars模型支持在KITTI格式数据集上训练，本项目使用的数据集地址为：

```shell
cd datasets/zw_1102_0914
```

数据集的目录结构组织如下：

```
└── kitti_dataset_root
    |—— training
        |—— label_2
            |—— 000001.txt
            |—— ...
        |—— calib
            |—— 000001.txt
            |—— ...
        |—— velodyne
            |—— 000001.bin
            |—— ...
    |—— ImageSets
        |—— test.txt
        |—— train.txt
        |—— trainval.txt
        |—— val.txt
```

数据集包含生成的数据增强所需的真值库，结构如下：

```
└── kitti_train_gt_database
    |—— anno_info_train.pkl
    |—— Car
        |—— 1102-day1_000000_Car_1.bin
        |—— ...
    |—— Pedestrian
        |—— 1102-day1_000000_Pedestrian_0.bin
        |—— ...
```

- 若需生成其他训练时数据增强所需的真值库，指令如下:

```
python tools/create_det_gt_database.py --dataset_name kitti --dataset_root /path/to/datasets --save_dir /path/to/datasets
```

`--dataset_root`指定KITTI数据集所在路径，`--save_dir`指定用于保存所生成的真值库的路径。

### <h3 id="22">训练</h3>
位于`Paddle3D/`目录下，执行：
```shell
python -m paddle.distributed.launch --gpus 0 \
    tools/train.py \
    --config configs/pointpillars/pointpillars_zw_car.yml \
    --save_interval 100 \
    --keep_checkpoint_max 100 \
    --save_dir outputs/pointpillars_zw_pedestrian_1102_0914 \
    --do_eval \
    --num_workers 8
```

训练脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项  |    默认值    |
|:--------------------|:-------------------------------|:------:|:---------:|
| gpus                | 使用的GPU编号                       |   是    |     -     |
| config              | 配置文件                           |   是    |     -     |
| save_dir            | 模型和visualdl日志文件的保存根路径          |   否    |  output   |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 |   否    |     2     |
| save_interval       | 模型保存的间隔步数                      |   否    |   1000    |
| do_eval             | 是否在保存模型时进行评估                   |   否    |     否     |
| log_interval        | 打印日志的间隔步数                      |   否    |    10     |
| keep_checkpoint_max | 最新模型保存个数                       |   否    |     5     |
| resume              | 是否从断点恢复训练                      |   否    |     否     |
| batch_size          | mini-batch大小（每张GPU）            |   否    | 在配置文件中指定  |
| iters               | 训练轮数                           |   否    | 在配置文件中指定  |
| learning_rate       | 学习率                            |   否    | 在配置文件中指定  |
| seed                | Paddle的全局随机种子值                         |   否    |   None    |

### <h3 id="23">评估</h3>

位于`Paddle3D/`目录下，执行：

```shell
python tools/evaluate.py \
    --config configs/pointpillars/pointpillars_zw_car.yml \
    --model /path/to/model.pdparams \
    --num_workers 8
```

评估脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项  |    默认值    |
|:--------------------|:-------------------------------|:------:|:---------:|
| config              | 配置文件                           |   是    |     -     |
| model               | 待评估模型路径                        |   是    |     -     |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 |   否    |     2     |
| batch_size          | mini-batch大小                   |   否    | 在配置文件中指定  |

### <h3 id="24">模型导出</h3>

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```shell
python tools/export.py \
    --config configs/pointpillars/pointpillars_zw_car.yml \
    --model /path/to/model.pdparams \
    --save_dir /path/to/output
```

模型导出脚本支持设置如下参数：

| 参数名         | 用途                                                                                                           | 是否必选项  |    默认值    |
|:------------|:-------------------------------------------------------------------------------------------------------------|:------:|:---------:|
| config      | 配置文件                                                                                                         |   是    |     -     |
| model       | 待导出模型参数`model.pdparams`路径                                                                                    |   是    |     -     |
| save_dir    | 保存导出模型的路径，`save_dir`下将会生成三个文件：`pointpillars.pdiparams `、`pointpillars.pdiparams.info`和`pointpillars.pdmodel` |   否    | `deploy`  |

### <h3 id="25">模型部署</h3>

#### Python 部署

**注意：目前PointPillars的仅支持使用GPU进行推理。**

运行命令参数说明如下：

| 参数名                 | 用途                                                                                    | 是否必选项 | 默认值 |
|:--------------------|:--------------------------------------------------------------------------------------|:------|:----|
| mdoel_file          | 导出模型的结构文件`pointpillars.pdmodel`所在路径                                                   | 是     | -   |
| params_file         | 导出模型的参数文件`pointpillars.pdiparams`所在路径                                                 | 是     | -   |
| lidar_file          | 待预测的点云所在路径                                                                            | 是     | -   |
| point_cloud_range   | 模型中将点云划分为柱体（pillars）时选取的点云范围，格式为`X_min Y_min Z_min X_max Y_Max Z_max`                 | 是     | -   |
| voxel_size          | 模型中将点云划分为柱体（pillars）时每个柱体的尺寸，格式为`X_size Y_size Z_size`                                | 是     | -   |
| max_points_in_voxel | 模型中将点云划分为柱体（pillars）时每个柱体包含点数量上限                                                      | 是     | -   |
| max_voxel_num       | 模型中将点云划分为柱体（pillars）时保留的柱体数量上限                                                        | 是     | -   |
| num_point_dim       | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4                    | 否     | 4   |
| use_trt             | 是否使用TensorRT进行加速                                                                      | 否     | 0   |
| trt_precision       | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16                                             | 否     | 0   |
| trt_use_static      | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成** | 否     | 0   |     |
| trt_static_dir      | 当trt_use_static设置为1时，保存优化信息的路径                                                        | 否     | -   |
| collect_shape_info  | 是否收集模型动态shape信息。默认0。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理**                     | 否     | 0   |     |
| dynamic_shape_file  | 保存模型动态shape信息的文件路径                                                                    | 否     | -   |

运行以下命令，执行预测：

```shell
python infer.py \
  --model_file /path/to/pointpillars.pdmodel \
  --params_file /path/to/pointpillars.pdiparams \
  --lidar_file /path/to/lidar.bin \
  --point_cloud_range 0 -39.68 -5 69.12 39.68 5 \
  --voxel_size .16 .16 10 \
  --max_points_in_voxel 32 \
  --max_voxel_num 40000
```

### <h3 id="25">推理结果可视化</h3>

运行以下命令，执行可视化：

```shell
python tools/pcd_vis_single_frame_demo.py   \
   --model_file outputs/pointpillars_zw_car_1102/best_model/pointpillars.pdmodel   \
   --params_file outputs/pointpillars_zw_car_1102/best_model/pointpillars.pdiparams   \
   --lidar_file datasets/zw-1102-618-all/zw-1102-618-train-merge/lidar_bin_xyzi   \
   --image_dir datasets/zw-1102-618-all/zw-1102-618-train-merge/camera/front  \
   --calib_file datasets/zw-1102-618-all/zw-1102-618-train-merge/KITTI_format/training/calib/1102-day1_000000.txt   \
   --point_cloud_range 0 -39.68 -5 69.12 39.68 5   \
   --voxel_size .16 .16 10   \
   --max_points_in_voxel 32   \
   --max_voxel_num 40000
```
