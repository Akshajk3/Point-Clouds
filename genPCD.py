import open3d as o3d
import numpy as np
import os

color_raw = o3d.io.read_image('outputs/color/color_IMG_1421.png')
depth_raw = o3d.io.read_image('outputs/depth/depth_IMG_1421.png')

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# mirror the image so it does not appear upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud('test.ply', pcd, format='auto', write_ascii=False)