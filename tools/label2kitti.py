from pathlib import Path
import json
import argparse
import math
import os
import numpy as np

parser = argparse.ArgumentParser(description='convert label to kitti format')
parser.add_argument('--src', type=str, default='datasets/zw_0830', help="source data folder")
parser.add_argument('--tgt', type=str, default='datasets/zw_0830_kitti_format', help="target folder")
parser.add_argument('--scenes', type=str, default='.*', help="")
parser.add_argument('--frames', type=str, default='.*', help="")

args = parser.parse_args()


def get_inv_matrix(file, v2c, rect):
    with open(file) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith(v2c), lines)][0]

        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        matrix = matrix + [0, 0, 0, 1]
        m = np.array(matrix)
        velo_to_cam = m.reshape([4, 4])

        trans = [x for x in filter(lambda s: s.startswith(rect), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        m = np.array(matrix).reshape(3, 3)

        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)

        rect = np.concatenate((m, np.expand_dims(np.array([0, 0, 0, 1]), 0)), axis=0)

        m = np.matmul(rect, velo_to_cam)

        return m


def get_detection_inv_matrix():
    return get_inv_matrix('datasets/zw_0830/load3-mid-speed_sync/calib/calib.txt',
                          "Tr_velo_to_cam", "R0_rect")

tgt_dir = Path(args.tgt)
label_files = [x for x in Path(args.src).rglob('*.json') if x.parent.stem == 'label']
for lf in label_files:
    with open(lf) as fin:
        label = json.load(fin)
    if 'objs' in label:
        label = label['objs']

    tgt_path = (tgt_dir / lf.relative_to(args.src)).with_suffix('.txt')
    tgt_path.parent.mkdir(exist_ok=True, parents=True)

    with open(tgt_path, 'w') as fout:
        for obj in label:

            x,y,z =obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']
            z -= obj['psr']['scale']['z'] / 2
            pos = np.array([float(x), float(y), float(z), 1]).T

            trans = get_detection_inv_matrix()
            trans_pos = np.matmul(trans, pos)

            print(pos)
            print(trans_pos)
            print(obj)
            line = "{} 0 0 0 0 0 0 0 {} {} {} {} {} {} {}\n".format(
                obj['obj_type'],
                obj['psr']['scale']['z'],  # h
                obj['psr']['scale']['y'],  # w
                obj['psr']['scale']['x'],  # l
                trans_pos[0],  # x
                trans_pos[1],  # y
                trans_pos[2],  # z
                -obj['psr']['rotation']['z'] - math.pi / 2,  # rotation_y
            )

            fout.write(line)
