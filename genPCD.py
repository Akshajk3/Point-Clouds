import open3d as o3d
import numpy as np
import os

color_raw = []
depth_raw = []

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

for filename in os.listdir('outputs/color'):
    filepath = os.path.join('outputs/color', filename)
    color_raw.append(filepath)

for filename in os.listdir('outputs/depth'):
    filepath = os.path.join('outputs/depth', filename)
    depth_raw.append(filepath)

final_pcd = o3d.geometry.PointCloud()

for color_path, depth_path in zip(color_raw, depth_raw):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    final_pcd += pcd

o3d.visualization.draw_geometries([final_pcd])
o3d.io.write_point_cloud('final.ply', final_pcd, format='auto', write_ascii=False)