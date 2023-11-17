import shutil
from pathlib import Path
import open3d as o3d
import numpy as np
import json
import math


def collect_unlabeled(out_dir):
    root_dir = Path('/workspace/SUSTechPOINTS/data')
    select_dict = {
        'load1-low-speed_sync':
            [4120, 4290, 4360, 4490, 4500, 4710, 4770, 4800, 4830, 4880, 4930, 4980, 5930, 6070, 6120, 7140, 7330, 7390,
             7420, 7780, 7820, 7830, 7840],
        'load2-low-speed_sync':
            [140, 500, 590, 640, 740, 770, 820, 940, 1120, 1170, 1200, 1350, 1400, 1470, 1520, 1580, 1630, 1850, 1920,
             2000, 2040, 2090, 2160, 2290, 2550, 3580, 3660, 4550, 4650, 11810, 11930],
        'load3-mid-speed_sync':
            [1050, 1180, 3890, 5750, 5800, 5860, 5910, 5960, 6010, 6050, 6150, 6250, 6350, 6400, 6450, 6520, 6550,
             12470, 12750],
        'PKU-Around_sync':
            [180, 770, 1030, 1090, 1150, 1250, 1550, 1740, 3850, 3950, 4160, 4360, 4500, 4650, 5000, 5100, 5600, 5850,
             6520, 9450],
        'PKU-Inside_sync':
            [6540, 6600, 6750, 7270],
        'Target-Tank_sync':
            [640, 650, 700, 800, 810, 820, 830],
        'Terrain_Cliff_Wall_sync':
            [1310, 1360, 1980, 2070, 2080, 2140, 2370],
        'Terrain-Trench_sync':
            [780],
        'Turnel_sync':
            [40, 150, 300, 450, 550, 750, 900, 1200, 1350, 1700, 1900, 2250, 2450, 2500]
    }
    out_dir = Path(out_dir)
    camera_dir = out_dir / 'camera' / 'front'
    label_dir = out_dir / 'label'
    lidar_dir = out_dir / 'lidar'
    camera_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    lidar_dir.mkdir(exist_ok=True, parents=True)
    for seq_name, id_list in select_dict.items():
        for x in id_list:
            pc_file = root_dir / seq_name / 'lidar' / f'{x:06d}.pcd'
            print(pc_file.exists())

            print(seq_name, pc_file)
            fname = f'{seq_name}_{pc_file.stem}'
            shutil.copy2(str(pc_file).replace('lidar', 'camera/front').replace('.pcd', '.png'),
                         camera_dir / f'{fname}.png')
            shutil.copy2(pc_file, lidar_dir / f'{fname}.pcd')


if __name__ == '__main__':
    out_dir = '/workspace/Paddle3D/datasets/zw_anno_0920'
    collect_unlabeled(out_dir)
