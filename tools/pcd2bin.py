import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


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

def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        type=str,
        default="/data2/dataset/zw-1102-618-all/night2/lidar"
    )
    parser.add_argument(
        "--bin_dir",
        help=".bin file path.",
        type=str,
        default="/data2/dataset/zw-1102-618-all/night2/lidar_bin_xyzi"
    )
    args = parser.parse_args()

    flist = list(Path(args.pcd_path).rglob('*.pcd'))
    for pcd_file in tqdm(flist):
        bin_file = Path(pcd_file).parent.parent / args.bin_dir / f'{pcd_file.stem}.bin'
        bin_file.parent.mkdir(exist_ok=True, parents=True)
        pcd2bin_pypcd(pcd_file, bin_file)

if __name__ == "__main__":
    main()