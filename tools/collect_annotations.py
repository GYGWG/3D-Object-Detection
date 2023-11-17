import shutil
from pathlib import Path
import open3d as o3d
import numpy as np
import json
import math

def collect_annotation(in_dir, out_dir):
    label_files = list(Path(in_dir).rglob('*.json'))
    label_files = [x for x in label_files if x.parent.stem == 'label']
    print(len(label_files), label_files[:5])
    out_dir = Path(out_dir)
    print(out_dir)
    camera_dir = out_dir / 'camera' / 'front'
    label_dir = out_dir / 'label'
    lidar_dir = out_dir / 'lidar'
    camera_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    lidar_dir.mkdir(exist_ok=True, parents=True)

    for lf in label_files:
        seq_name = lf.parent.parent.stem
        if seq_name == 'example':
            continue
        fname = f'{seq_name}_{lf.stem}'

        if (label_dir / f'{fname}.json').exists():
            continue
        if not Path(str(lf).replace('label', 'lidar').replace('.json', '.pcd')).exists():
            continue
        shutil.copy2(lf, label_dir / f'{fname}.json')
        shutil.copy2(str(lf).replace('label', 'camera/front').replace('.json', '.png'), camera_dir / f'{fname}.png')
        shutil.copy2(str(lf).replace('label', 'lidar').replace('.json', '.pcd'), lidar_dir / f'{fname}.pcd')


def convert_to_kitti(root_dir):
    CLASS_LIST = ['Car', 'Pedestrian', 'Misc']
    def pcd2bin_pypcd(pcd_file, bin_file):
        # pip install pypcd-imp
        from pypcd_imp import pypcd
        pc = pypcd.PointCloud.from_path(pcd_file)
        np_x = pc.pc_data['x'].astype(np.float32)
        np_y = pc.pc_data['y'].astype(np.float32)
        np_z = pc.pc_data['z'].astype(np.float32)
        np_i = pc.pc_data['intensity'].astype(np.float32) / 255
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
        # remove nan
        mask = np.isnan(np_x) + np.isnan(np_y) + np.isnan(np_z) + np.isnan(np_i)
        points_32 = points_32[mask == 0]
        points_32.tofile(bin_file)

    def label_to_kitti(label_file, calib_file, out_path):
        with open(calib_file) as f:
            lines = f.readlines()
            trans = [x for x in filter(lambda s: s.startswith('Tr_velo_to_cam'), lines)][0]

            matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
            matrix = matrix + [0, 0, 0, 1]
            m = np.array(matrix)
            velo_to_cam = m.reshape([4, 4])

            trans = [x for x in filter(lambda s: s.startswith('R0_rect'), lines)][0]
            matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
            m = np.array(matrix).reshape(3, 3)

            m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)

            rect = np.concatenate((m, np.expand_dims(np.array([0, 0, 0, 1]), 0)), axis=0)
            m = np.matmul(rect, velo_to_cam)


        with open(label_file) as fin:
            label = json.load(fin)
        if 'objs' in label:
            label = label['objs']

        with open(out_path, 'w') as fout:
            for obj in label:
                x, y, z = obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']
                z -= obj['psr']['scale']['z'] / 2
                pos = np.array([float(x), float(y), float(z), 1]).T

                trans = m
                trans_pos = np.matmul(trans, pos)

                cname = obj['obj_type']
                if cname == 'Van' or cname == 'DontCare':
                    cname = 'Misc'
                if cname not in CLASS_LIST:
                    print(cname)

                line = "{} 0.0 0.0 0.0 0.0 0.0 50.0 50.0 {} {} {} {} {} {} {}\n".format(
                    cname,
                    obj['psr']['scale']['z'],  # h
                    obj['psr']['scale']['y'],  # w
                    obj['psr']['scale']['x'],  # l
                    trans_pos[0],  # x
                    trans_pos[1],  # y
                    trans_pos[2],  # z
                    -obj['psr']['rotation']['z'] - math.pi / 2,  # rotation_y
                )
                fout.write(line)

    root_dir = Path(root_dir)
    out_dir = root_dir / 'KITTI_format'
    out_lidar_dir = out_dir / 'training' / 'velodyne'
    out_calib_dir = out_dir / 'training' / 'calib'
    out_label_dir = out_dir / 'training' / 'label_2'
    out_set_dir = out_dir / 'ImageSets'
    out_set_dir.mkdir(exist_ok=True, parents=True)
    out_lidar_dir.mkdir(exist_ok=True, parents=True)
    out_calib_dir.mkdir(exist_ok=True, parents=True)
    out_label_dir.mkdir(exist_ok=True, parents=True)
    label_files = list((root_dir / 'label').rglob('*.json'))

    with open(root_dir / 'calib' / 'camera' / 'front.json') as f:
        calib_dict = json.load(f)
        intrinsic = calib_dict['intrinsic']
        intrinsic = np.hstack((np.array(calib_dict['intrinsic']).reshape(3, 3), np.zeros([3, 1]))).reshape(-1).tolist()
        line = ' '.join(list(map(str, intrinsic)))
        line = f'P0: {line}\n' \
               f'P1: {line}\n' \
               f'P2: {line}\n' \
               f'P3: {line}\n'
        line += 'R0_rect: ' + ' '.join(list(map(str, np.eye(3).reshape(-1).tolist()))) + '\n'
        line += 'Tr_velo_to_cam: ' + ' '.join(list(map(str, calib_dict['extrinsic'][:12]))) + '\n'
        line += 'Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

    with open(out_set_dir / 'train.txt', 'w') as fset:
        for lf in label_files:
            fset.write(f'{lf.stem}\n')

    for lf in label_files:
        lidar_file = str(lf).replace('label', 'lidar').replace('.json', '.pcd')
        pcd2bin_pypcd(lidar_file, out_lidar_dir / f'{lf.stem}.bin')
        calib_file = out_calib_dir / f'{lf.stem}.txt'
        with open(calib_file, 'w') as cf:
            cf.write(line)
        label_to_kitti(lf, calib_file, out_label_dir / f'{lf.stem}.txt')


if __name__ == '__main__':
    in_dir = '/data3/zw-1102-618-all/zw-1102-618-train'
    out_dir = '/data3/zw-1102-618-all/zw-1102-618-train-merge'
    # collect_annotation(in_dir, out_dir)

    convert_to_kitti(out_dir)
