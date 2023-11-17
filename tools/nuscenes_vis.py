from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/data2/dataset/NuScenes/v1.0-trainval05_blobs/', verbose=True)

# nusc.list_scenes()

# my_scene = nusc.scene[0]
# print(my_scene)

nusc.render_sample_data("c422cb7d5fef4d68ad1aeeeab40e145c")

