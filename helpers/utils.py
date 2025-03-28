import imp
import math
from tkinter import E
import numpy as np
import pybullet as p
import cv2
import copy
import open3d as o3d
import open3d_plus as o3dp
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.constants import WORKSPACE_LIMITS, PIXEL_SIZE, NUM_POINTS, FIX_MAX_ERR


reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': 'models/graspnet/logs/log_rs/checkpoint.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 15,
    'mask_thresh': 0.5
}

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[px, py] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[px, py, c] = colors[:, c]
    return heightmap, colormap

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = config["intrinsics"]
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)

    return heightmaps, colormaps

def get_fuse_heightmaps(obs, configs, bounds, pixel_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = reconstruct_heightmaps(
        obs["color"], obs["depth"], configs, bounds, pixel_size
    )
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.

    return cmap, hmap

def get_true_heightmap(env):
    """Get RGB-D orthographic heightmaps and segmentation masks in simulation."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(env.oracle_cams[0]) 

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.oracle_cams, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask

def get_camera_heightmap(env, config):
    """Get RGB-D orthographic heightmaps and segmentation masks in simulation."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(config) 

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], [config], env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask

def get_heightmap_from_real_image(color, depth, segm, env):
    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.camera.configs, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.uint8(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask

def process_pcds(pcds, reconstruction_config):
    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config['voxel_size']
        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        if len(np.asarray(income_pcd.points)) == 0 or len(np.asarray(pcd.points)) == 0:
            return None, None
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd = o3dp.merge_pcds([pcd, income_pcd])
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd

def get_topdown_image_pointcloud(env, max_depth=35):
    config = env.oracle_cams[0]
    color, depth, _ = env.render_camera(config)

    intrinsics = config["intrinsics"]
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)

    depth = depth / 1000 * 35 

    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    xyz = np.float32([px, py, depth]).transpose(1, 2, 0)

    # apply depth mask
    mask = (xyz[..., -1] < max_depth)
    xyz = xyz[mask]
    
    # scale depth -- [999.9, 1000] scales to [0, 27] then to [8, 35] 
    # xyz[..., -1] = (xyz[..., -1] - 999.9) / (1000 - 999.9) * 27 + 8

    # NaN check
    mask = np.logical_not(np.isnan(np.sum(xyz, axis=-1)))
    xyz = xyz[mask]

    # visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
    # o3d.visualization.draw_geometries([pcd])

    return color, pcd

def get_fuse_pointcloud(env):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, _ = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd

def get_obj_pcd_from_mask(env, obj_id):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, mask = env.render_camera(config)
        # get padding images only containing object
        bbox_color = np.zeros(color.shape).astype(np.uint8)
        bbox_color[mask == obj_id] = color[mask == obj_id]
        bbox_depth = np.ones(depth.shape).astype(np.uint8) * 1000.00055
        bbox_depth[mask == obj_id] = depth[mask == obj_id]

        xyz = get_pointcloud(bbox_depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = bbox_color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    # if all pcds are with 0 points, then result None, else remove pcds with 0 points
    raw_pcds = pcds[:]
    for pcd in raw_pcds:
        if len(np.asarray(pcd.points)) == 0:
            pcds.remove(pcd)
    
    if len(pcds) >= 2:
        _, fuse_pcd = process_pcds(pcds, reconstruction_config)
        if fuse_pcd is None:
            return None
        else:
            # visualization
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # o3d.visualization.draw_geometries([fuse_pcd, frame])
            return fuse_pcd
    
    elif len(pcds) == 1:
        return pcds[0]
    else:
        return None

def get_obj_pcd(pcd, bbox_pixel, z_range=None):
    # translate bbox pixels to bbox positions
    bbox_pos = []
    for i in range(len(bbox_pixel)): # [y0, y1, x0, x1] 
        if i <= 1:
            bbox_pos.append(bbox_pixel[i] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]) # (y0, x0) -> (y1, x1)
        else:
            bbox_pos.append(bbox_pixel[i] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0])
    bbox_pos = np.array(bbox_pos)
    # Filter out 3D points that are outside of the predefined bounds.
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    ix = (points[Ellipsis, 0] >= bbox_pos[0]) & (points[Ellipsis, 0] < bbox_pos[1])
    iy = (points[Ellipsis, 1] >= bbox_pos[2]) & (points[Ellipsis, 1] < bbox_pos[3])
    if z_range is None:
        iz = (points[Ellipsis, 2] >= 0.0001) & (points[Ellipsis, 2] < WORKSPACE_LIMITS[2, 1])
    else:
        iz = (points[Ellipsis, 2] >= z_range[0]) & (points[Ellipsis, 2] < z_range[1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]
    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    # visualization
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    o3d.visualization.draw_geometries([pcd, frame])
    
    return pcd

def pose_estimation_icp(source, target, voxel_size=0.005):
    # drawt the initial pcds
    # draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # get an initial estimated pose by fast icp
    import time
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)
    # get a more precise estimated pose by icp
    source.estimate_normals()
    target.estimate_normals()
    result_icp_from_fast = refine_registration(source, target, result_fast,
                                    voxel_size)
    print(result_icp_from_fast)
    # draw_registration_result(source, target, result_icp_from_fast.transformation)   

    # get an initial estimated pose by ransac
    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print(result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)
    # get a more precise estimated pose by icp
    source.estimate_normals()
    target.estimate_normals()
    result_icp_from_ransac = refine_registration(source, target, result_ransac,
                                    voxel_size)
    print(result_icp_from_ransac)
    # draw_registration_result(source, target, result_icp_from_ransac.transformation) 

    # fitness comparison
    results = np.array([result_fast, result_icp_from_fast, result_ransac, result_icp_from_ransac])
    fitness = np.array([result_fast.fitness, result_icp_from_fast.fitness, result_ransac.fitness, result_icp_from_ransac.fitness])
    best_result_id = np.argmax(fitness)
    best_transformation = results[best_result_id].transformation
    
    return best_transformation

def preprocess_point_cloud(pcd, voxel_size):
    print("Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print("Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print("Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("RANSAC registration on downsampled point clouds.")
    print("Since the downsampling voxel size is %.3f," % voxel_size)
    print("we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    print("Point-to-plane ICP registration is applied on original point")
    print("clouds to refine the alignment. This time we use a strict")
    print("distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def get_true_bboxs_with_ids(env, color_image, depth_image, mask_image):
    # get mask of all objects
    bbox_images = []
    bbox_positions = []
    bbox_obj_ids = []
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            if stats[:-1].shape[0] > 1:
                size = stats[:-1, -1]
                bbox = stats[:-1][np.argmax(size)]
            else:
                bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # visualization
            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
            thickness = 1 # Line thickness of 1 px 
            mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)

            cv2.imwrite('mask_bboxs.png', mask_bboxs)

            # add padding=2
            upper_left = np.max((y0-2, 0))
            lower_left = np.min((y1+2, 224))
            upper_right = np.max((x0-2, 0))
            lower_right = np.min((x1+2, 224))
            bbox_image = color_image[upper_left:lower_left, upper_right:lower_right]
            # bbox_image = color_image[(y0-2):(y1+2), (x0-2):(x1+2)]
            bbox_images.append(bbox_image)
            
            pixel_x = (x0 + x1) // 2
            pixel_y = (y0 + y1) // 2
            bbox_pos = [
                pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
            ]
            bbox_positions.append(bbox_pos)
            bbox_obj_ids.append(obj_id)

    return bbox_images, bbox_positions, bbox_obj_ids

def get_true_bboxs(env, color_image, depth_image, mask_image):
    # get mask of all objects
    bbox_images = []
    bbox_positions = []
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            if stats[:-1].shape[0] > 1:
                size = stats[:-1, -1]
                bbox = stats[:-1][np.argmax(size)]
            else:
                bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # visualization
            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
            thickness = 1 # Line thickness of 1 px 
            mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)

            cv2.imwrite('mask_bboxs.png', mask_bboxs)

            # add padding=2
            upper_left = np.max((y0-2, 0))
            lower_left = np.min((y1+2, 224))
            upper_right = np.max((x0-2, 0))
            lower_right = np.min((x1+2, 224))
            bbox_image = color_image[upper_left:lower_left, upper_right:lower_right]
            # bbox_image = color_image[(y0-2):(y1+2), (x0-2):(x1+2)]
            bbox_images.append(bbox_image)
            
            pixel_x = (x0 + x1) // 2
            pixel_y = (y0 + y1) // 2
            bbox_pos = [
                pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
            ]
            bbox_positions.append(bbox_pos)

    return bbox_images, bbox_positions

def get_true_obj_bboxs(color_image, depth_image, mask_image, obj_id):
    mask = np.zeros(mask_image.shape).astype(np.uint8)
    mask[mask_image == obj_id] = 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:,4].argsort()]
    if stats[:-1].shape[0] > 0:
        if stats[:-1].shape[0] > 1:
            size = stats[:-1, -1]
            bbox = stats[:-1][np.argmax(size)]
        else:
            bbox = stats[:-1][0]
        # for bbox
        # |(y0, x0)         |   
        # |                 |
        # |                 |
        # |         (y1, x1)|
        x0, y0 = bbox[0], bbox[1]
        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        # visualization
        start_point, end_point = (x0, y0), (x1, y1)
        color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
        thickness = 1 # Line thickness of 1 px 
        mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
        mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
        cv2.imwrite('mask_bboxs.png', mask_bboxs)

        # add padding=2
        upper_left = np.max((y0-2, 0))
        lower_left = np.min((y1+2, 224))
        upper_right = np.max((x0-2, 0))
        lower_right = np.min((x1+2, 224))
        bbox_image = color_image[upper_left:lower_left, upper_right:lower_right]
        # bbox_image = color_image[(y0-2):(y1+2), (x0-2):(x1+2)]
        
        pixel_x = (x0 + x1) // 2
        pixel_y = (y0 + y1) // 2
        bbox_pos = [
            pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
            depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
        ]
    else:
        bbox_image = None
        bbox_pos = None

    return bbox_image, bbox_pos

# Note, if an object is dropped first but fully occupied by others, this function will not catch this object's padding image
def get_true_bbox_padding_whole_image(env, color_image, mask_image): 
    padding_image = np.zeros(color_image.shape).astype(np.uint8)
    # get mask of all objects
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            if stats[:-1].shape[0] > 1:
                size = stats[:-1, -1]
                bbox = stats[:-1][np.argmax(size)]
            else:
                bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            
            padding_image[y0:y1, x0:x1] = color_image[y0:y1, x0:x1]
            cv2.imwrite('padding_image.png', padding_image)

    return padding_image

# Note, if an object is dropped first but fully occupied by others, this function will not catch this object's padding image
def get_true_bbox_padding_images(env, color_image, mask_image): 
    # get mask of all objects
    bbox_padding_images = []
    bbox_pixels = []
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            if stats[:-1].shape[0] > 1:
                size = stats[:-1, -1]
                bbox = stats[:-1][np.argmax(size)]
            else:
                bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            bbox_padding_image = np.zeros(color_image.shape).astype(np.uint8)
            bbox_padding_image[y0:y1, x0:x1] = color_image[y0:y1, x0:x1]
            # bbox_padding_image = cv2.cvtColor(bbox_padding_image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('bbox_padding_image.png', bbox_padding_image)

            bbox_pixel = [y0, y1, x0, x1]

            bbox_padding_images.append(bbox_padding_image)
            bbox_pixels.append(bbox_pixel)

    return bbox_padding_images, bbox_pixels

def get_true_bbox_padding_image(color_image, mask_image, obj_id):
    mask = np.zeros(mask_image.shape).astype(np.uint8)
    mask[mask_image == obj_id] = 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:,4].argsort()]
    if stats[:-1].shape[0] > 0:
        if stats[:-1].shape[0] > 1:
            size = stats[:-1, -1]
            bbox = stats[:-1][np.argmax(size)]
        else:
            bbox = stats[:-1][0]
        # for bbox
        # |(y0, x0)         |   
        # |                 |
        # |                 |
        # |         (y1, x1)|
        x0, y0 = bbox[0], bbox[1]
        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        bbox_padding_image = np.zeros(color_image.shape).astype(np.uint8)
        bbox_padding_image[y0:y1, x0:x1] = color_image[y0:y1, x0:x1]

        bbox_padding_image = cv2.cvtColor(bbox_padding_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('bbox_padding_image.png', bbox_padding_image)

        bbox_pixel = [y0, y1, x0, x1]
    else:
        bbox_padding_image = None
        bbox_pixel = None

    return bbox_padding_image, bbox_pixel

def execute_place(env, grasp_action, grasped_obj_id, match_pred, cur_depth_image, before_obj_poses, \
                    target_mask_image, target_bbox_pixels, target_bbox_positions, target_obj_poses, target_obj_pcds, \
                    evaluate_pose=True, use_pose_gt=True, with_arm=False, bottom_background=False):

    print('match pred', match_pred)
    target_bbox_pixel = target_bbox_pixels[match_pred]
    # depth image of the target mask
    obj_id = env.target_obj_ids[match_pred]
    target_obj_mask = np.zeros(target_mask_image.shape).astype(np.uint8)
    target_obj_mask[target_mask_image == obj_id] = 1
    target_obj_depth_image = cur_depth_image * target_obj_mask
    # depth image of the target object zone
    # target_obj_depth_image = cur_depth_image[target_bbox_pixel[0]:target_bbox_pixel[1], target_bbox_pixel[2]:target_bbox_pixel[3]]
    # if target position is occupied (except occupied by itself), then move to an unoccupied buffer
    # !!!! Modify the bound when change the background!!! depth will change!!!
    if not bottom_background:
        buffer_flag = target_obj_depth_image.max() > 5e-5 and target_obj_depth_image.max() < 0.2 # !!workspace.obj
    else:
        buffer_flag = target_obj_depth_image.max() > 0.011 and target_obj_depth_image.max() < 0.25  # !!bottom.obj
    
    if buffer_flag:
        print('Move to buffer!')
        # find a non-occupied positon for buffer excluding target zones
        free_space = np.ones_like(cur_depth_image)
        for target_bbox_pixel in target_bbox_pixels:
            free_space[target_bbox_pixel[0]-20:target_bbox_pixel[1]+20, target_bbox_pixel[2]-20:target_bbox_pixel[3]+20] = 0
        
        if not bottom_background:
            free_space[np.where(cur_depth_image > 5e-5)] = 0 # !!workspace.obj
        else:
            free_space[np.where(cur_depth_image > 0.01004873)] = 0 # !!bottom.obj
        # remove edge zones, workspace: 224*224
        free_space[:20, :] = 0
        free_space[204:, :] = 0
        free_space[:, :20] = 0
        free_space[:, 204:] = 0

        free_space_idxs = np.where(free_space)
        if free_space_idxs[0].shape[0] == 0:
            print('No free space for buffer! Place back!')
            pos = before_obj_poses[grasped_obj_id][:3, 3]
        else:
            pixel_idx = np.random.choice(free_space_idxs[0].shape[0])
            pixel_y = free_space_idxs[0][pixel_idx]
            pixel_x = free_space_idxs[1][pixel_idx]
            pos = np.array([
                pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                cur_depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
            ])
        if with_arm:
            #### place with arm ####
            rot = grasp_action[3:]
            pose = np.hstack((pos, rot))
            place_success = env.place(pose)
        else:
            #### place with object reorientation ####
            pose = before_obj_poses[grasped_obj_id]
            pose[:3, 3] = pos
            place_success = env.place_object_pose(pose, grasped_obj_id)
        
    else:
        print('Move to target!')
        if not evaluate_pose:
            # place with target position
            rot = grasp_action[3:]
            pos = target_bbox_positions[match_pred]
            pose = np.hstack((pos, rot))
            place_success = env.place(pose)
        else:
            # place with target pose
            if use_pose_gt:
                obj_id = env.target_obj_ids[match_pred]
                if with_arm:
                    ### place with arm ####
                    place_success = env.place_pose_offset(target_obj_poses[obj_id], grasped_obj_id)  
                else:
                    ### place with object reorientation ####
                    place_success = env.place_object_pose(target_obj_poses[obj_id], grasped_obj_id)
            
            else:

                # two-stage placement, choose a more accurate transformation
                target_obj_pcd = target_obj_pcds[match_pred]
                cur_obj_pcd = get_obj_pcd_from_mask(env, grasped_obj_id)
                if cur_obj_pcd is None:
                    # object is moved out of workspace, move it back to its original position
                    print('Move out of the workspace!! Place back!!')
                    rot = grasp_action[3:]
                    pos = grasp_action[:3]
                    pos[2] -= 0.2
                    pose = np.hstack((pos, rot))
                    place_success = env.place(pose)
                else:
                    relative_pose_prediction = pose_estimation_icp(cur_obj_pcd, target_obj_pcd)     
                    print(relative_pose_prediction)
                    place_success = env.place_delta_pose(relative_pose_prediction, intermediate=True)

                cur_obj_pcd = get_obj_pcd_from_mask(env, grasped_obj_id)
                if cur_obj_pcd is None:
                    # object is moved out of workspace, move it back to its original position
                    print('Move out of the workspace!! Place back!!')
                    rot = grasp_action[3:]
                    pos = grasp_action[:3]
                    pos[2] -= 0.2
                    pose = np.hstack((pos, rot))
                    place_success = env.place(pose)

                else:
                    relative_pose_prediction = pose_estimation_icp(cur_obj_pcd, target_obj_pcd)     
                    print(relative_pose_prediction)     
                    place_success = env.place_delta_pose(relative_pose_prediction)  
    
    return place_success

def relabel_mask(env, mask_image):
    assert env.target_obj_id != -1
    num_obj = 50
    for i in np.unique(mask_image):
        if i == env.target_obj_id:
            mask_image[mask_image == i] = 255
        elif i in env.obj_ids["rigid"]:
            mask_image[mask_image == i] = num_obj
            num_obj += 10
        else:
            mask_image[mask_image == i] = 0
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def relabel_mask_real(masks):
    """Assume the target object is labeled to 255"""
    mask_image = np.zeros_like(masks[0], dtype=np.uint8)
    num_obj = 50
    for idx, mask in enumerate(masks):
        if idx == 0:
            mask_image[mask == 255] = 255
        else:
            mask_image[mask == 255] = num_obj
            num_obj += 10
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def rotate(image, angle, is_mask=False):
    """Rotate an image using cv2, counterclockwise in degrees"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if is_mask:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated

def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def bbox_preprocess(bbox_images, bbox_positions, n_px):
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    remain_bboxes = []
    remain_bbox_positions = []
    remain_bboxes_ = []
    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 5 and bbox_images[i].shape[1] >= 5:
            remain_bboxes.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_positions.append(bbox_positions[i])
    print('Remaining bbox number', len(remain_bboxes))
    bboxes = None
    for remain_bbox in remain_bboxes:
        remain_bbox = Image.fromarray(remain_bbox)
        # padding
        w, h = remain_bbox.size
        if w >= h:
            remain_bbox_ = Image.new(mode='RGB', size=(w,w))
            remain_bbox_.paste(remain_bbox, box=(0, (w-h)//2))
        else:
            remain_bbox_ = Image.new(mode='RGB', size=(h,h))
            remain_bbox_.paste(remain_bbox, box=((h-w)//2, 0))
        remain_bbox_ = transform(remain_bbox_)

        remain_bboxes_.append(remain_bbox_.cpu().numpy().transpose(1, 2, 0))

        remain_bbox_ = remain_bbox_.unsqueeze(0) 
        if bboxes == None:
            bboxes = remain_bbox_
        else:
            bboxes = torch.cat((bboxes, remain_bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    pos_bboxes = None
    for bbox_pos in remain_bbox_positions:
        bbox_pos = torch.from_numpy(np.array(bbox_pos))
        bbox_pos = bbox_pos.unsqueeze(0)
        if pos_bboxes == None:
            pos_bboxes = bbox_pos
        else:
            pos_bboxes = torch.cat((pos_bboxes, bbox_pos), dim=0) # shape = [n_obj, pos_dim]
    if pos_bboxes != None:
        pos_bboxes = pos_bboxes.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_obj, pos_dim]

    return remain_bboxes, remain_bboxes_, bboxes, pos_bboxes

def target_bbox_preprocess(bbox_images, n_px):
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    bboxes = None
    for bbox in bbox_images:
        bbox = Image.fromarray(bbox)
        # padding
        w,h = bbox.size
        if w >= h:
            bbox_ = Image.new(mode='RGB', size=(w,w))
            bbox_.paste(bbox, box=(0, (w-h)//2))
        else:
            bbox_ = Image.new(mode='RGB', size=(h,h))
            bbox_.paste(bbox, box=((h-w)//2, 0))
        bbox_ = transform(bbox_)
        bbox_ = bbox_.unsqueeze(0) 
        if bboxes == None:
            bboxes = bbox_
        else:
            bboxes = torch.cat((bboxes, bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    return bboxes

# Preprocess of model input
def preprocess(bbox_images, bbox_positions, grasp_pose_set, n_px):
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    remain_bboxes = []
    remain_bbox_positions = []
    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 15 and bbox_images[i].shape[1] >= 15:
            remain_bboxes.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_positions.append(bbox_positions[i])
    print('Remaining bbox number', len(remain_bboxes))
    bboxes = None
    for remain_bbox in remain_bboxes:
        remain_bbox = Image.fromarray(remain_bbox)
        # padding
        w,h = remain_bbox.size
        if w >= h:
            remain_bbox_ = Image.new(mode='RGB', size=(w,w))
            remain_bbox_.paste(remain_bbox, box=(0, (w-h)//2))
        else:
            remain_bbox_ = Image.new(mode='RGB', size=(h,h))
            remain_bbox_.paste(remain_bbox, box=((h-w)//2, 0))
        remain_bbox_ = transform(remain_bbox_)

        remain_bbox_ = remain_bbox_.unsqueeze(0)
        if bboxes == None:
            bboxes = remain_bbox_
        else:
            bboxes = torch.cat((bboxes, remain_bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    pos_bboxes = None
    for bbox_pos in remain_bbox_positions:
        bbox_pos = torch.from_numpy(np.array(bbox_pos))
        bbox_pos = bbox_pos.unsqueeze(0)
        if pos_bboxes == None:
            pos_bboxes = bbox_pos
        else:
            pos_bboxes = torch.cat((pos_bboxes, bbox_pos), dim=0) # shape = [n_obj, pos_dim]
    if pos_bboxes != None:
        pos_bboxes = pos_bboxes.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_obj, pos_dim]
    
    grasps = None
    for grasp in grasp_pose_set:
        grasp = torch.from_numpy(grasp)
        grasp = grasp.unsqueeze(0)
        if grasps == None:
            grasps = grasp
        else:
            grasps = torch.cat((grasps, grasp), dim=0) # shape = [n_grasp, grasp_dim]
    grasps = grasps.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    return remain_bboxes, bboxes, pos_bboxes, grasps

def preprocess_with_ids(bbox_images, bbox_positions, bbox_obj_ids, grasp_pose_set, n_px):
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    remain_bboxes = []
    remain_bbox_positions = []
    remain_bbox_obj_ids = []
    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 5 and bbox_images[i].shape[1] >= 5:
            remain_bboxes.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_positions.append(bbox_positions[i])
            remain_bbox_obj_ids.append(bbox_obj_ids[i])
    print('Remaining bbox number', len(remain_bboxes))
    bboxes = None
    for remain_bbox in remain_bboxes:
        remain_bbox = Image.fromarray(remain_bbox)
        # padding
        w,h = remain_bbox.size
        if w >= h:
            remain_bbox_ = Image.new(mode='RGB', size=(w,w))
            remain_bbox_.paste(remain_bbox, box=(0, (w-h)//2))
        else:
            remain_bbox_ = Image.new(mode='RGB', size=(h,h))
            remain_bbox_.paste(remain_bbox, box=((h-w)//2, 0))
        remain_bbox_ = transform(remain_bbox_)

        remain_bbox_ = remain_bbox_.unsqueeze(0)
        if bboxes == None:
            bboxes = remain_bbox_
        else:
            bboxes = torch.cat((bboxes, remain_bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    pos_bboxes = None
    for bbox_pos in remain_bbox_positions:
        bbox_pos = torch.from_numpy(np.array(bbox_pos))
        bbox_pos = bbox_pos.unsqueeze(0)
        if pos_bboxes == None:
            pos_bboxes = bbox_pos
        else:
            pos_bboxes = torch.cat((pos_bboxes, bbox_pos), dim=0) # shape = [n_obj, pos_dim]
    if pos_bboxes != None:
        pos_bboxes = pos_bboxes.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_obj, pos_dim]
    
    grasps = None
    if len(grasp_pose_set) > 0:
        for grasp in grasp_pose_set:
            # transfer to rotation vector
            # rot_vec = R.from_quat(grasp[-4:]).as_rotvec()
            # grasp[-3:] = rot_vec
            # grasp = grasp[:6]
            grasp = torch.from_numpy(grasp)
            grasp = grasp.unsqueeze(0)
            if grasps == None:
                grasps = grasp
            else:
                grasps = torch.cat((grasps, grasp), dim=0) # shape = [n_grasp, grasp_dim]
        grasps = grasps.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    return remain_bboxes, remain_bbox_obj_ids, remain_bbox_positions, bboxes, pos_bboxes, grasps

def plot_match(bboxes, target_bboxes, match, score):
    plt.figure()
    for i in range(len(target_bboxes)):
        ax = plt.subplot(3, len(bboxes), i+1)
        plt.imshow(target_bboxes[i])
        plt.xticks([])
        plt.yticks([])
    for i in range(len(bboxes)):
        # bboxes[i] = cv2.cvtColor(bboxes[i], cv2.COLOR_RGB2BGR)
        ax = plt.subplot(3, len(bboxes), len(bboxes)+i+1)
        plt.imshow(bboxes[i])
        plt.xticks([])
        plt.yticks([])
        ax = plt.subplot(3, len(bboxes), 2*len(bboxes)+i+1)
        plt.imshow(target_bboxes[match[i]])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(str(score[i]), fontsize=10)
    plt.show()

def plot_obj_match(bbox, target_bboxes, score):
    plt.figure()
    ax = plt.subplot(2, len(target_bboxes), 1)
    plt.imshow(bbox)
    plt.xticks([])
    plt.yticks([])
    for i in range(len(target_bboxes)):
        # bboxes[i] = cv2.cvtColor(bboxes[i], cv2.COLOR_RGB2BGR)
        ax = plt.subplot(2, len(target_bboxes), len(target_bboxes)+i+1)
        plt.imshow(target_bboxes[i])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(str(score[i]), fontsize=10)
    plt.show()

def plot_probs(text, bboxes, probs):
    plt.figure()
    plt.suptitle(text)
    for i in range(len(bboxes)):
        # bboxes[i] = cv2.cvtColor(bboxes[i], cv2.COLOR_RGB2BGR)
        ax = plt.subplot(1, len(bboxes), i+1)
        plt.imshow(bboxes[i])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(str(probs[0][i]), fontsize=10)
    plt.show()
    
def plot_prob_bar(probs):
    fig, ax = plt.subplots()
    # ax.set_yticks([])
    # ax.set_xticks([])
    # im = ax.imshow(probs, cmap="YlGnBu", interpolation='nearest')
    # plt.colorbar(im)
    import seaborn as sns
    sns.heatmap(probs, cmap="YlGnBu", annot=True, cbar=True, xticklabels=False, yticklabels=False, square=True)

    fig.tight_layout()
    plt.show()

def plot_attnmap(attn_map):
    fig, ax = plt.subplots()
    ax.set_yticks([])
    # ax.set_yticks(range(attn_map.shape[0]))
    ax.set_xticks([])
    im = ax.imshow(attn_map, cmap="YlGnBu", interpolation='nearest')
    plt.colorbar(im)

    fig.tight_layout()
    plt.show()

def load_object_points(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(NUM_POINTS)
    points = np.asarray(pcd.points)
    return points

def draw_line(p1, p2, a = 1e-3, color = np.array((0.0,1.0,0.0))):
    '''Get a open3d.geometry.TriangleMesh of a line

    Args:
        p1(np.array): the first point.
        p2(np.array): the second point.
        a(float): the length of the square of the bottom face.
    Returns:
        open3d.geometry.TriangleMesh: the line.
    '''
    vertex_1 = o3d.geometry.TriangleMesh.create_box(1.5 * a,1.5 * a,1.5 * a)
    vertex_1.translate(p1 - np.array((0.75 * a, 0.75 * a, 0.75 * a)))
    vertex_1.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array((1.0,0,0)), (8, 1))) 
    vertex_2 = o3d.geometry.TriangleMesh.create_box(1.5 * a,1.5 * a,1.5 * a)
    vertex_2.translate(p2 - np.array((0.75 * a, 0.75 * a, 0.75 * a)))
    vertex_2.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array((1.0,0,0)), (8, 1))) 
    d = np.linalg.norm(p1 - p2)
    v1 = (p2 - p1) / d
    v2 = np.cross(np.array((0,0,1.0)), v1)
    v3 = np.cross(v1, v2)
    R = np.stack((v3, v2, v1)).astype(np.float64).T
    box = o3d.geometry.TriangleMesh.create_box(width = a, height = a, depth = d)
    box = box.translate(np.array((-a / 2, -a / 2, 0)))
    trans_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array((0,0,0,1))))
    # print('trans_matrix:{}'.format(trans_matrix))
    box = box.transform(trans_matrix)
    box = box.translate(p1)
    box.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1))) 
    return box, vertex_1, vertex_2

def compute_add(points_1, points_2, debug=False):
    dump_pcd = o3d.geometry.PointCloud()
    dump_pcd.points = o3d.utility.Vector3dVector(points_1)
    dump_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points_2)
    target_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    lines = []
    for i in range(0, NUM_POINTS, 10):
        box, v1, v2 = draw_line(points_1[i], points_2[i])
        lines.append(box)
    if debug:
        o3d.visualization.draw_geometries([frame, target_pcd, dump_pcd] + lines)
    if not points_1.shape == points_2.shape:
        raise ValueError('The shape of points must be the same')
    add = np.mean(np.linalg.norm((points_1 - points_2), axis = 1))
    return add
    
def get_closest_point(points_1, points_2):
    def norm(t):
        return np.sqrt(np.sum(t * t, axis=-1))
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    points_1 = points_1[:, np.newaxis]
    points_2 = points_2[np.newaxis, :]
    dist = norm(points_1 - points_2)
    indices = np.argmin(dist, axis = -1)
    min_dist = dist[np.array(list(range(points_1.shape[0]))), indices]
    return min_dist, indices

def get_farest_point(points_1, points_2):
    def norm(t):
        return np.sqrt(np.sum(t * t, axis=-1))
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    points_1 = points_1[:, np.newaxis]
    points_2 = points_2[np.newaxis, :]
    dist = norm(points_1 - points_2)
    indices = np.argmax(dist, axis = -1)
    max_dist = dist[np.array(list(range(points_1.shape[0]))), indices]
    return max_dist, indices

def compute_adds(points_1, points_2, debug=False):
    # each of point in points_1 to best point in points_2
    min_dist, indices = get_closest_point(points_1, points_2)
    dump_pcd = o3d.geometry.PointCloud()
    dump_pcd.points = o3d.utility.Vector3dVector(points_1)
    dump_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points_2)
    target_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    lines = []
    for i in range(0, NUM_POINTS, 10):
        box, v1, v2 = draw_line(points_1[i], points_2[indices[i]])
        lines.append(box)
    if debug:
        o3d.visualization.draw_geometries([frame, target_pcd, dump_pcd] + lines)
    return np.mean(min_dist)
    
def get_diameter_from_points(points):
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=float)
    assert points.shape[1] == 3, 'points have to have 3-elements'

    covariance_matrix = np.cov(points, y=None, rowvar=0, bias=1)  
    _, eigen_vectors = np.linalg.eigh(covariance_matrix)  

    trans_matrix = eigen_vectors
    rotate_matrix = np.transpose(trans_matrix)

    # 由原始点转换到AABB，求中心和长宽高
    p_primes = np.array([np.dot(p, trans_matrix) for p in points])
    min_p = np.min(p_primes, axis=0)
    max_p = np.max(p_primes, axis=0)
    # center = np.dot((min_p + max_p) / 2.0, rotate_matrix)
    # extents = (max_p - min_p)

    return np.linalg.norm(max_p - min_p)

def calculate_score(source_pose, target_pose, mesh_path, symmetric_metric, debug=False):
    mesh_points = load_object_points(mesh_path)
    source_points = np.matmul(source_pose['mat'], mesh_points.T).T + source_pose['t']
    target_points = np.matmul(target_pose['mat'], mesh_points.T).T + target_pose['t']
    model_diameter = get_diameter_from_points(target_points)

    max_bound = FIX_MAX_ERR
    if not symmetric_metric:
        score = compute_add(source_points, target_points, debug)
        object_score = min(score, max_bound)
    else:
        score = compute_adds(source_points, target_points, debug)
        object_score = min(score, max_bound)
        
    return object_score, model_diameter

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotm2euler(R):

    assert isRotm(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert isRotm(R)

    if (
        (abs(R[0][1] - R[1][0]) < epsilon)
        and (abs(R[0][2] - R[2][0]) < epsilon)
        and (abs(R[1][2] - R[2][1]) < epsilon)
    ):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if (
            (abs(R[0][1] + R[1][0]) < epsilon2)
            and (abs(R[0][2] + R[2][0]) < epsilon2)
            and (abs(R[1][2] + R[2][1]) < epsilon2)
            and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)
        ):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if (xx > yy) and (xx > zz):  # R[0][0] is the largest diagonal term
            if xx < epsilon:
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:  # R[1][1] is the largest diagonal term
            if yy < epsilon:
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if zz < epsilon:
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2])
        + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0])
        + (R[1][0] - R[0][1]) * (R[1][0] - R[0][1])
    )  # used to normalise
    if abs(s) < 0.001:
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]

def is_intersection(ax, ay, px, py, x1, y1, x2, y2):
    # Points: A(ax, ay), P(px, py)
    # Rectangle: 
    # |(x1, y1)         |   
    # |                 |
    # |                 |
    # |         (x2, y2)|
    ax, ay, px, py = float(ax), float(ay), float (px), float (py)
    x1, y1, x2, y2 = float (x1), float (y1), float (x2), float (y2)
    if py == ay:
        if py == y1 or py == y2:
            return True
    else:
        # intersection point of line and y = y1
        sx = (y1 - ay) * (px - ax) / (py - ay) + ax
        if sx >= x1 and sx <= x2:
            return True
        # intersection point of line and y = y2
        xx = (y2 - ay) * (px - ax) / (py - ay) + ax
        if xx >= x1 and sx <= x2:
            return True
    if px == ax:
        if px == x1 or px == x2:
            return True
    else:
        # intersection point of line and x = x1
        zy = (py - ay) * (x1 - ax) / (px - ax) + ay
        if zy >= y1 and zy <= y2:
            return True
        # intersection point of line and x = x2
        yy = (py - ay) * (x2 - ax) / (px - ax) + ay
        if yy >= y1 and yy <= y2:
            return True
    return False
         