import os
import numpy as np
import open3d as o3d
import sys


class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.P1 = dict_calib['P1'].reshape(3, 4)
        self.P2 = dict_calib['P2'].reshape(3, 4)
        self.P3 = dict_calib['P3'].reshape(3, 4)
        self.R0_rect = dict_calib['R0_rect'].reshape(3, 3)
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape(3, 4)


class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        lines = content.split()
        lines = list(filter(lambda x: len(x), lines))
        mode = sys.argv[1]

        if mode == 'label':
            self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(
                lines[3])
            self.bbox = [lines[4], lines[5], lines[6], lines[7]]
            self.bbox = np.array([float(x) for x in self.bbox])
            self.dimensions = [lines[8], lines[9], lines[10]]
            self.dimensions = np.array([float(x) for x in self.dimensions])
            self.location = [lines[11], lines[12], lines[13]]
            self.location = np.array([float(x) for x in self.location])
            self.rotation_y = float(lines[14])
            # self.score = float(lines[15])
        elif mode == 'pred':
            self.name, self.truncated, self.occluded, self.alpha = 'Car', 1.0, 0.0, 0.0
            self.bbox = [0, 0, 50, 50]
            self.bbox = np.array([float(x) for x in self.bbox])
            self.dimensions = [lines[-4], lines[-3], lines[-2]]
            self.dimensions = np.array([float(x) for x in self.dimensions])
            self.location = [lines[-7], lines[-6], lines[-5]]
            self.location = np.array([float(x) for x in self.location])
            self.rotation_y = float(lines[-1])


def draw_3d_box(vis, points, name):
    position = points
    points_box = np.transpose(position)

    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
    # import pdb
    # pdb.set_trace()
    # lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    #                       [8, 9], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    # lines_box = np.array([[0, 0]])
    # for i in range(8):
    #     for j in range(8):
    #         lines_box = np.append(lines_box, [[i,j]], axis = 0)
    # import pdb
    # pdb.set_trace()
    # if name == 'Car':
    if name == 1:
        colors = np.array([[1., 0., 0.] for j in range(len(lines_box))])
    # elif name == 'Pedestrian':
    elif name == 2:
        colors = np.array([[1., 1., 0.] for j in range(len(lines_box))])
    elif name == 3:
        colors = np.array([[1., 1., 1.] for j in range(len(lines_box))])
    else:
        colors = np.array([[1., 0., 1.] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # breakpoint()
    print("==============")
    print(np.asarray(line_set.points))

    vis.update_geometry(line_set)
    param = o3d.io.read_pinhole_camera_parameters('BirdView.json')

    ctr = vis.get_view_control()
    vis.add_geometry(line_set)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_geometry(line_set)
    vis.update_renderer()


def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R


def parse_info(lidar_path, calib_path, label_path):
    data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])

    with open(calib_path) as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x) and x != '\n', lines))
    dict_calib = {}
    for line in lines:
        key, value = line.split(":")
        dict_calib[key] = np.array([float(x) for x in value.split()])
    calib = Calib(dict_calib)

    with open(label_path, 'r') as f:
        mode = sys.argv[1]
        lines = f.readlines()
        if mode == 'pred':
            lines = list(filter(lambda x: len(x) and x != '\n' and float(x.split()[1]) > 0.0, lines[:1]))
        elif mode == 'label':
            lines = list(filter(lambda x: len(x) and x != '\n', lines))
    obj = [Object3d(x) for x in lines]
    return point_cloud, calib, obj


def viz_pc(lidar_path, label_path, calib_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1440, height=1080)
    render_option = vis.get_render_option()
    render_option.line_width = 10
    render_option.point_size = 2
    render_option.background_color = np.array([0, 0, 0])
    # render_option.color
    # import cv2
    # img_path = lidar_path.replace('.bin', '.png').replace('lidar_bin', 'camera/front')
    # img = cv2.imread(img_path)
    # print(img.shape)

    point_cloud, calib, obj = parse_info(lidar_path, calib_path, label_path)
    point_cloud.paint_uniform_color([0, 255 / 255, 255 / 255])
    vis.add_geometry(point_cloud)

    for obj_index in range(len(obj)):
        if obj[obj_index].name in ['Car', 'Pedestrian', 'Cyclist']:
            R = rot_y(obj[obj_index].rotation_y)
            h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]

            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            # x = [0, 0, -l ,-l, 0, 0, -l ,-l]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            # z = [0, -w, -w, 0, 0, -w, -w, 0]

            corner_3d = np.vstack([x, y, z])
            corner_3d = np.dot(R, corner_3d)

            corner_3d[0, :] += obj[obj_index].location[0]
            corner_3d[1, :] += obj[obj_index].location[1]
            corner_3d[2, :] += obj[obj_index].location[2]

            # tr_cam2img = np.array([925.9414012936553, 0, 521.1386244584028,
            #                        0, 926.7738802299361, 413.9576387269848,
            #                        0, 0, 1]).reshape(3,3)
            #
            # corner_2d = np.dot(tr_cam2img, corner_3d)
            # corner_2d /= corner_2d[-1, :]
            # for x, y, _ in corner_2d.T:
            #     cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), -1)
            # cv2.imwrite('img.png', img)
            # print(calib.R0_rect)
            # breakpoint()

            mode = sys.argv[1]
            if mode == 'label':
                corner_3d = np.dot(np.linalg.inv(calib.R0_rect), corner_3d)
                corner_3d = np.vstack((corner_3d, np.ones((1, corner_3d.shape[-1]))))
                m = np.vstack([calib.Tr_velo_to_cam, np.zeros((1, 4))])
                m[-1][-1] = 1
                m = np.linalg.inv(m)
                Y = np.dot(m[:3], corner_3d)  # (3, 8)
            else:
                Y = corner_3d

            draw_3d_box(vis, Y, obj[obj_index].name)
    vis.run()


def get_rotation(heading):
    cos = np.cos(heading)
    sin = np.sin(heading)
    return np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])


def get_3d_box_corners(box):
    center_x, center_y, center_z, width, length, height, heading = box
    rotation = get_rotation(heading)  # (3, 3)
    translation = [center_x, center_y, center_z]

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    corners_3d = np.array([
        [l2, w2, -h2], [-l2, w2, -h2], [-l2, -w2, -h2], [l2, -w2, -h2],
        [l2, w2, h2], [-l2, w2, h2], [-l2, -w2, h2], [l2, -w2, h2]
    ])  # (8, 3)

    corners_3d = np.dot(rotation, np.transpose(corners_3d))  # (3, 8)
    corners_3d[0, :] += translation[0]
    corners_3d[1, :] += translation[1]
    corners_3d[2, :] += translation[2]
    return corners_3d  # (3, 8)


def viz_waymo(pkl_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1440, height=1080)
    render_option = vis.get_render_option()
    render_option.line_width = 10
    render_option.point_size = 4
    render_option.background_color = np.array([0, 0, 0])

    import pickle
    with open(pkl_path, 'rb') as f:
        frame = pickle.load(f)
    data = frame['data']
    labels = frame['labels']
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])
    point_cloud.paint_uniform_color([0, 255 / 255, 255 / 255])
    vis.add_geometry(point_cloud)

    for label in labels:
        box, type = label['box'], label['type']
        corner_3d = get_3d_box_corners(box)
        draw_3d_box(vis, corner_3d, type)
        # import pdb;pdb.set_trace()
    vis.run()


def main():
    # lidar_root = 'datasets/KITTI/training/velodyne'
    # for lidar_nm in sorted(os.listdir(lidar_root)):
    #     if lidar_nm.endswith('.DS_Store'):
    #         continue
    #     print(lidar_nm)
    #     lidar_pt = os.path.join(lidar_root, lidar_nm)
    #     label_pt = lidar_pt.replace('velodyne', 'label_2').replace('.bin', '.txt')
    #     calib_pt = label_pt.replace('label_2', 'calib')
    #     viz_pc(lidar_pt, label_pt, calib_pt)
    # pkl_path = 'data.pkl'
    # viz_waymo(pkl_path)
    mode, file = sys.argv[1], int(sys.argv[2])
    name = '%06d' % file
    # lidar_pt = f'datasets/KITTI/training/velodyne/{name}.bin'
    lidar_pt = f'datasets/ZW/training/velodyne/{name}.bin'
    if mode == 'label':
        # label_pt = f'datasets/ZW/training/label_2/{name}.txt'
        label_pt = f'datasets/zw_anno_0914/KITTI_format/training/label_2/load1-low-speed_sync_{name}.txt'
        # label_pt = f'datasets/KITTI/training/label_2/{name}.txt'
    elif mode == 'pred':
        label_pt = f'result/load1/{name}.txt'
        # label_pt = f'result/KITTI_all/{name}.txt'
    # calib_pt = f'datasets/KITTI/training/calib/{name}.txt'
    calib_pt = f'datasets/ZW/training/calib/{name}.txt'
    viz_pc(lidar_pt, label_pt, calib_pt)


if __name__ == '__main__':
    main()
