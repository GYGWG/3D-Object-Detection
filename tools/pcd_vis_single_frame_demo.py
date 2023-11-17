# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor
from paddle3d.ops.iou3d_nms import nms_gpu
from tools.vis_utils import preprocess, Calibration, show_lidar_with_boxes, show_bev_with_boxes, show_image_with_boxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        '--image_dir', type=str, help='The lidar path.', default=None)
    parser.add_argument(
        '--calib_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        '--results_dir', type=str, help='The lidar path.', default=None)
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        "--point_cloud_range",
        dest='point_cloud_range',
        nargs='+',
        help="Range of point cloud for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--voxel_size",
        dest='voxel_size',
        nargs='+',
        help="Size of voxels for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--max_points_in_voxel",
        type=int,
        default=100,
        help="Maximum number of points in a voxel.")
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.4,
        help="Score threshold for visualization.")
    parser.add_argument(
        "--max_voxel_num",
        type=int,
        default=12000,
        help="Maximum number of voxels.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        type=int,
        default=0,
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")
    parser.add_argument(
        "--collect_shape_info",
        type=int,
        default=0,
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="Path of a dynamic shape file for tensorrt.")
    parser.add_argument(
        "--infer_object_file",
        type=str,
        default="",
        help="Path of a inference file.")

    return parser.parse_args()


def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=10,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    return predictor


def run(predictor, voxels, coords, num_points_per_voxel):
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        if name == "voxels":
            input_tensor.reshape(voxels.shape)
            input_tensor.copy_from_cpu(voxels.copy())
        elif name == "coords":
            input_tensor.reshape(coords.shape)
            input_tensor.copy_from_cpu(coords.copy())
        elif name == "num_points_per_voxel":
            input_tensor.reshape(num_points_per_voxel.shape)
            input_tensor.copy_from_cpu(num_points_per_voxel.copy())

    # do the inference
    predictor.run()

    # get out data from output tensor
    output_names = predictor.get_output_names()

    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        if i == 0:
            box3d_lidar = output_tensor.copy_to_cpu()
        elif i == 1:
            label_preds = output_tensor.copy_to_cpu()
        elif i == 2:
            scores = output_tensor.copy_to_cpu()
    return box3d_lidar, label_preds, scores


if __name__ == '__main__':
    args = parse_args()

    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)

    from pathlib import Path
    if Path(args.lidar_file).is_file():
        lidar_file_list = [args.lidar_file]
    else:
        lidar_file_list = list(Path(args.lidar_file).glob('*.bin'))
        lidar_file_list.sort()
    if args.image_dir is None or Path(args.image_dir).is_file():
        image_file_list = [args.image_dir] * len(lidar_file_list)
    else:
        image_file_list = list(Path(args.image_dir).glob('*.png'))
        image_file_list.sort()

    results_dir = args.results_dir
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True)

    for lidar_file, image_file in zip(lidar_file_list, image_file_list):
        print(lidar_file, image_file)
        voxels, coords, num_points_per_voxel = preprocess(
            lidar_file, args.num_point_dim, args.point_cloud_range,
            args.voxel_size, args.max_points_in_voxel, args.max_voxel_num)
        box3d_lidar, label_preds, scores = run(predictor, voxels, coords,
                                               num_points_per_voxel)

        # results = np.hstack([box3d_lidar, label_preds.reshape(-1, 1), scores.reshape(-1, 1)])
        # np.savetxt(results_dir / f'{lidar_file.stem}.txt', results, fmt='%.3f', delimiter='\t')

        scan = np.fromfile(lidar_file, dtype=np.float32)
        pc_velo = scan.reshape((-1, 4))

        # Obtain calibration information about Kitti
        calib = Calibration(args.calib_file)
        # Plot box in bev
        bev_im = show_bev_with_boxes(pc_velo, box3d_lidar, scores, 0.12)
        cv2.imshow('bev', bev_im)
        cam_im = cv2.imread(str(image_file))
        # box_im = show_image_with_boxes(cam_im, box3d_lidar, scores, calib, args.score_thr)
        box_im = show_image_with_boxes(cam_im, box3d_lidar, scores, calib, 0.12)
        cv2.imshow('cam', box_im)
        cv2.waitKey(0)

    # Plot box in lidar cloud
    #show_lidar_with_boxes(pc_velo, box3d_lidar, scores, label_preds, args.score_thr)

    # args = parse_args()

    # inference_file = args.infer_object_file
    # with open(inference_file, "r") as f:
    #     lines = f.read().split('\n')
    # scores = np.array([])
    # label_preds = np.empty((0, 1), int)
    # box3d_lidar = np.empty((0, 7), float)
    # for line in lines:
    #     if line:
    #         score = line.split(' ')[1]
    #         scores = np.append(scores, float(score))
    #         label = line.split(' ')[3]
    #         label_preds = np.append(label_preds, int(label))
    #         bbox = [float(line.split(' ')[-9]), float(line.split(' ')[-8]), float(line.split(' ')[-7]),
    #                 float(line.split(' ')[-6]), float(line.split(' ')[-5]), float(line.split(' ')[-4]),
    #                 float(line.split(' ')[-1])]
    #         box3d_lidar = np.append(box3d_lidar, [bbox], axis=0)
    #
    # scan = np.fromfile(args.lidar_file, dtype=np.float32)
    # pc_velo = scan.reshape((-1, 4))
    #
    # # Obtain calibration information about Kitti
    # calib = Calibration(args.calib_file)
    #
    # # Plot box in lidar cloud
    # show_lidar_with_boxes(pc_velo, box3d_lidar, scores, label_preds)
