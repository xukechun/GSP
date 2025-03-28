import numpy as np
import open3d as o3d
import open3d_plus as o3dp
from scipy.spatial.transform import Rotation as R
import copy

from models.graspnet.graspnet_baseline import GraspNetBaseLine
from helpers.utils import graspnet_config
from graspnetAPI import GraspGroup

class Graspnet:
    def __init__(self):
        self.config = graspnet_config
        self.graspnet_baseline = GraspNetBaseLine(checkpoint_path = self.config['graspnet_checkpoint_path'])

    def compute_grasp_pose(self, full_pcd):
        points, _ = o3dp.pcd2array(full_pcd)
        grasp_pcd = copy.deepcopy(full_pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-points)

        # generating grasp poses.
        gg = self.graspnet_baseline.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.config['refine_approach_dist']
        gg = self.graspnet_baseline.collision_detection(gg, points)

        # all the returned result in 'world' frame. 'gg' using 'graspnet' gripper frame.
        return gg

    def assign_grasp_pose(self, gg, object_poses):
        grasp_poses = dict()
        grasp_pose_set = []
        dist_thresh = self.config['dist_thresh']
        # - dist_thresh: float of the minimum distance from the grasp pose center to the object center. The unit is millimeter.
        angle_thresh = self.config['angle_thresh']
        # - angle_thresh:
        #             /|
        #            / |
        #           /--|
        #          /   |
        #         /    |
        # Angle should be smaller than this angle

        # gg: GraspGroup in 'world' frame of 'graspnet' gripper frame.
        # x is the approaching direction.
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        # move the center to the eelink frame
        # Note that here is rs[:,:,0] before
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)

        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        remain_gg_array = []
        angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))

        if object_poses is None:
            grasp_poses['all'] = []
            # if np.sum(angle_mask) >= self.config['mask_thresh']:
            #     scores = scores[angle_mask]
            #     ts = ts[angle_mask]
            #     eelink_rs = eelink_rs[angle_mask]
            #     rs = rs[angle_mask]
            #     gg = gg[angle_mask]

            for i in range(len(gg)):
                remain_gg_array.append(gg[i].grasp_array)
                grasp_rotation_matrix = eelink_rs[i]
                if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                    # print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                    grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                # else:
                #     print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                
                grasp_pose = np.zeros(7)
                grasp_pose[:3] = [ts[i][0], ts[i][1], ts[i][2]]
                r = R.from_matrix(grasp_rotation_matrix)
                grasp_pose[-4:] = r.as_quat()

                grasp_poses['all'].append(grasp_pose)
                grasp_pose_set.append(grasp_pose)

        else:
            # min_dist: np.array of the minimum distance to any object(must > dist_thresh)
            min_dists = np.inf * np.ones((len(rs)))

            # min_object_ids: np.array of the id of the nearest object.
            min_object_ids = -1 * np.ones(shape = (len(rs)), dtype = np.int32)
            # first round to find the object that each grasp belongs to.
            for i, object_name in enumerate(object_poses.keys()):
                object_pose = object_poses[object_name]

                dists = np.linalg.norm(ts - object_pose[:3,3], axis=1)
                object_mask = np.logical_and(dists < min_dists, dists < dist_thresh)

                min_object_ids[object_mask] = i
                min_dists[object_mask] = dists[object_mask]
            
            
            # second round to calculate the parameters
            for i, object_name in enumerate(object_poses.keys()):
                obj_id_mask = (min_object_ids == i)
                add_angle_mask = (obj_id_mask & angle_mask)
                # For safety and planning difficulty reason, grasp pose with small angle with gravity direction will be accept.
                # if no grasp pose is available within the safe cone. grasp pose with the smallest angle will be used without
                # considering the angle.
                if np.sum(add_angle_mask) < self.config['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                    mask = obj_id_mask
                    sorting_method = 'angle'
                else:
                    mask = add_angle_mask
                    sorting_method = 'score'
                # print(f'{object_name} using sorting method: {sorting_method}, mask num: {np.sum(mask)}')
                i_scores = scores[mask]
                i_ts = ts[mask]
                i_eelink_rs = eelink_rs[mask]
                i_rs = rs[mask]
                i_gg = gg[mask]

                if np.sum(mask) < self.config['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                    # ungraspable
                    grasp_poses[object_name] = None
                else:
                    grasp_poses[object_name] = []
                    for i in range(len(i_gg)):
                        remain_gg_array.append(i_gg[i].grasp_array)
                        grasp_rotation_matrix = i_eelink_rs[i]
                        if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                            # print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                            grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                        # else:
                        #     print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                        
                        grasp_pose = np.zeros(7)
                        grasp_pose[:3] = [i_ts[i][0], i_ts[i][1], i_ts[i][2]]
                        r = R.from_matrix(grasp_rotation_matrix)
                        grasp_pose[-4:] = r.as_quat()

                        grasp_poses[object_name].append(grasp_pose)
                        grasp_pose_set.append(grasp_pose)

        remain_gg = GraspGroup(np.array(remain_gg_array))      

        return grasp_pose_set, grasp_poses, remain_gg


    def grasp_detection(self, full_pcd, object_poses=None, apply_nms=False, visualize=False):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry infomation is used in this implementation.

        There are mainly three steps.
        - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
        - Generating objects 6d poses by mainly icp matching.
        - Generating grasping poses by graspnet-baseline.

        Args:
            object_list(list): strings of object names.
            pose_method: string of the 6d pose estimation method, "icp" or "superglue".
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''

        # generate grasping poses in a scene
        gg = self.compute_grasp_pose(full_pcd)

        grasp_pose_set, grasp_pose_dict, remain_gg = self.assign_grasp_pose(gg, object_poses)
        
        if apply_nms:
            nms_gg = remain_gg.nms(translation_thresh = 0.05, rotation_thresh = 10 / 180.0 * 3.1416)

        if visualize:
            # visualization
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])
            o3d.visualization.draw_geometries([frame, full_pcd, *remain_gg.to_open3d_geometry_list()])
            if apply_nms:
                o3d.visualization.draw_geometries([frame, full_pcd, *nms_gg.to_open3d_geometry_list()])

        return grasp_pose_set, grasp_pose_dict, remain_gg


    def assign_grasp_pose_w_score(self, gg, object_poses):
        dist_thresh = self.config['dist_thresh']
        # - dist_thresh: float of the minimum distance from the grasp pose center to the object center. The unit is millimeter.
        angle_thresh = self.config['angle_thresh']
        # - angle_thresh:
        #             /|
        #            / |
        #           /--|
        #          /   |
        #         /    |
        # Angle should be smaller than this angle

        # gg: GraspGroup in 'world' frame of 'graspnet' gripper frame.
        # x is the approaching direction.
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        # move the center to the eelink frame
        # Note that here is rs[:,:,0] before
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)

        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        if object_poses is None:
            # if np.sum(angle_mask) >= self.config['mask_thresh']:
            #     scores = scores[angle_mask]
            #     ts = ts[angle_mask]
            #     eelink_rs = eelink_rs[angle_mask]
            #     rs = rs[angle_mask]
            #     gg = gg[angle_mask]

            from IPython import embed; embed()
            best_grasp_id = np.argmax(scores)
            best_gg_array = gg[int(best_grasp_id)].grasp_array   
            best_gg = GraspGroup(np.array(best_gg_array))          

            grasp_rotation_matrix = eelink_rs[best_grasp_id]
            if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                # print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
            # else:
            #     print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
            
            best_grasp_pose = np.zeros(7)
            best_grasp_pose[:3] = [ts[best_grasp_id][0], ts[best_grasp_id][1], ts[best_grasp_id][2]]
            r = R.from_matrix(grasp_rotation_matrix)
            best_grasp_pose[-4:] = r.as_quat()
        
        else:
            # min_dist: np.array of the minimum distance to any object(must > dist_thresh)
            min_dists = np.inf * np.ones((len(rs)))

            # min_object_ids: np.array of the id of the nearest object.
            min_object_ids = -1 * np.ones(shape = (len(rs)), dtype = np.int32)

            # first round to find the object that each grasp belongs to.
            angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))
            
            for i, object_name in enumerate(object_poses.keys()):
                object_pose = object_poses[object_name]

                dists = np.linalg.norm(ts - object_pose[:3,3], axis=1)
                object_mask = np.logical_and(dists < min_dists, dists < dist_thresh)

                min_object_ids[object_mask] = i
                min_dists[object_mask] = dists[object_mask]
            
            # second round to calculate the parameters
            for i, object_name in enumerate(object_poses.keys()):
                obj_id_mask = (min_object_ids == i)
                add_angle_mask = (obj_id_mask & angle_mask)
                # For safety and planning difficulty reason, grasp pose with small angle with gravity direction will be accept.
                # if no grasp pose is available within the safe cone. grasp pose with the smallest angle will be used without
                # considering the angle.
                if np.sum(add_angle_mask) < self.config['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                    mask = obj_id_mask
                    sorting_method = 'angle'
                else:
                    mask = add_angle_mask
                    sorting_method = 'score'
                # print(f'{object_name} using sorting method: {sorting_method}, mask num: {np.sum(mask)}')
                i_scores = scores[mask]
                i_ts = ts[mask]
                i_eelink_rs = eelink_rs[mask]
                i_rs = rs[mask]
                i_gg = gg[mask]

                if np.sum(mask) < self.config['mask_thresh']: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                    # ungraspable
                    best_grasp_pose = None
                    best_gg = None
                else:
                    best_grasp_id = np.argmax(i_scores)
                    best_gg_array = i_gg[int(best_grasp_id)].grasp_array 
                    best_gg = GraspGroup(np.array(best_gg_array))     

                    grasp_rotation_matrix = i_eelink_rs[best_grasp_id]
                    if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                        # print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                        grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                    # else:
                    #     print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                    
                    best_grasp_pose = np.zeros(7)
                    best_grasp_pose[:3] = [i_ts[best_grasp_id][0], i_ts[best_grasp_id][1], i_ts[best_grasp_id][2]]
                    r = R.from_matrix(grasp_rotation_matrix)                    
                    best_grasp_pose[-4:] = r.as_quat()
                        
        return best_grasp_pose, best_gg


    def single_grasp_detection(self, full_pcd, object_poses=None, visualize=False):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry infomation is used in this implementation.

        There are mainly three steps.
        - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
        - Generating objects 6d poses by mainly icp matching.
        - Generating grasping poses by graspnet-baseline.

        Args:
            object_list(list): strings of object names.
            pose_method: string of the 6d pose estimation method, "icp" or "superglue".
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''

        # generate grasping poses in a scene
        gg = self.compute_grasp_pose(full_pcd)
        best_grasp_pose, best_gg = self.assign_grasp_pose_w_score(gg, object_poses)

        # visualization
        if visualize:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])
            o3d.visualization.draw_geometries([frame, full_pcd, best_gg])

        return best_grasp_pose, best_gg