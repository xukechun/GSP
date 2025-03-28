from ast import keyword
from cProfile import label
import time
import glob
import os
from tkinter.messagebox import NO
import pybullet as pb
import pybullet_planning as pp
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import random
from operator import itemgetter
from scipy.spatial.transform import Rotation as R
from tomlkit import key
import env.cameras as cameras
from env.constants import PIXEL_SIZE, WORKSPACE_LIMITS, OBJECT_SYMMETRIC_MAP
import helpers.utils as utils

from pybullet_planning import wait_for_duration, link_from_name, get_disabled_collisions, get_movable_joints, set_joint_positions, \
                            plan_joint_motion, set_joint_positions, inverse_kinematics_helper

class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.novel_agent_cams = cameras.RealSenseD435.NEW_CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Start PyBullet.
        # pb = bullet_client.BulletClient(connection_mode=pb.GUI, "options=opengl2")
        # self._client_id = pb._client
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)

        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        info = (pos, rot, dim)
        return info    

    def get_target_id(self):
        return self.target_obj_ids

    def remove_target_id(self, obj_id):
        self.target_obj_ids.remove(obj_id)
        self.obj_ids["rigid"].remove(obj_id)
        pb.removeBody(obj_id)

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)
        pb.removeBody(obj_id)

    def save_objects(self):
        """Save states of all rigid objects. If this is unstable, could use saveBullet."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = pb.getBasePositionAndOrientation(obj)
            linVel, angVel = pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects. If this is unstable, could use restoreState along with saveBullet."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            pb.resetBasePositionAndOrientation(obj, pos, orn)
            pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = pb.loadURDF(
            "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )

        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = pb.loadURDF(
            "assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)


        # Load planning UR5e
        # self.ur5e_planning = pb.loadURDF(
        #     "assets/ur5e/ur5e.urdf", basePosition=(0, 1, 0), useFixedBase=True,
        # )
        # self.ur5e_planning_joints = []
        # for i in range(pb.getNumJoints(self.ur5e_planning)):
        #     info = pb.getJointInfo(self.ur5e_planning, i)
        #     joint_id = info[0]
        #     joint_name = info[1].decode("utf-8")
        #     joint_type = info[2]
        #     if joint_name == "ee_fixed_joint":
        #         self.ur5e_planning_ee_id = joint_id
        #     if joint_type == pb.JOINT_REVOLUTE:
        #         self.ur5e_planning_joints.append(joint_id)
        # pb.enableJointForceTorqueSensor(self.ur5e_planning, self.ur5e_planning_ee_id, 1)

        self.robot_collision_disabled_link_names = [('base_link', 'shoulder_link'),
            ('ee_link', 'wrist_1_link'), ('eide_link', 'wrist_2_link'),
            ('ee_link', 'wrist_3_link'), ('forearm_link', 'upper_arm_link'),
            ('forearm_link', 'wrist_1_link'), ('shoulder_link', 'upper_arm_link'),
            ('wrist_1_link', 'wrist_2_link'), ('wrist_1_link', 'wrist_3_link'),
            ('wrist_2_link', 'wrist_3_link')]

        self.robot_collision_links = get_disabled_collisions(self.ur5e, self.robot_collision_disabled_link_names)
        # each joint of robot are assigned with interger in pybullet
        self.robot_ik_joints = get_movable_joints(self.ur5e)[:6]
        # self.tool_attach_link_name = 'ee_link'
        # self.tool_attach_link = link_from_name(self.ur5e, self.tool_attach_link_name)

        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def get_obj_reward(self, obj_id, target_obj_dones, compute_pose_score=False):
        """Get step reward.

        Args:
            None.

        Returns:
            reward, done
        """
    
        pos_error = 0
        pose_error = 0
        punish = 0
        done = False

        # !!! map target object ids to current object ids !!!
        target_obj_ids = [4 + int(i) for i in range(len(self.target_obj_ids))]
        if obj_id in target_obj_ids:
            target_obj_ind = target_obj_ids.index(obj_id)
            if target_obj_ind in target_obj_dones.keys():
                punish = 1.5

            current_object_pose = dict()
            current_object_pose['mat'] = self.get_true_object_pose(obj_id)[:3, :3]
            current_object_pose['t'] = self.get_true_object_pose(obj_id)[:3, 3]
            target_obj_id = self.target_obj_ids[target_obj_ind]
            target_object_pose = self.target_obj_poses[target_obj_id]['pose']
            pos_error = np.linalg.norm(np.array(current_object_pose['t'])-np.array(target_object_pose['t']))
            
            if not compute_pose_score:
                # In baselines, they consider an episode to be complete when all target objects are placed within 5 cm from their goal position 
                if pos_error < 0.05: # 0.1 # pose_errors < 0.1 * len(self.target_obj_ids)
                    done = True
            else:
                mesh_file = self.target_obj_poses[target_obj_id]['mesh_file']
                symmetric_metric = self.target_obj_poses[target_obj_id]['symmetric']
                pose_error, model_diameter = utils.calculate_score(current_object_pose, target_object_pose, mesh_file, symmetric_metric, debug=False)
                # In pose estimation, they consider a pose estimation successful if adds < 0.1 * model diameter
                if pose_error < 0.35 * model_diameter:
                    done = True

        else:
            pos, _, _ = self.obj_info(obj_id)
            if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                done = True
        
        print(f"Position errors: {pos_error}.")
        # reward = done - pos_error
        reward = done - punish

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()

        if not compute_pose_score:
            return reward, done
        else:
            return reward, done, pos_error, pose_error

    def get_reward(self, obj_id, match_pred, match_gt, use_match_gt):
        """Get step reward.

        Args:
            None.

        Returns:
            reward, done
        """
        # compare desired poses and current poses of target objects
        # pose_errors = 0
        # for obj_id in self.target_obj_ids: # Note that here target object inds maybe different from the initialization, those out of workspace have been removed.
        #     mesh_file = self.target_obj_poses[obj_id]['mesh_file']
        #     current_object_pose = dict()
        #     current_object_pose['mat'] = self.get_true_object_pose(obj_id)[:3, :3]
        #     current_object_pose['t'] = self.get_true_object_pose(obj_id)[:3, 3]

        #     target_object_pose = self.target_obj_poses[obj_id]['pose']
        #     symmetric_metric = self.target_obj_poses[obj_id]['symmetric']

        #     trans_error = np.linalg.norm(np.array(current_object_pose['t'])-np.array(target_object_pose['t']))
        #     pose_error = utils.calculate_score(current_object_pose, target_object_pose, mesh_file, symmetric_metric, debug=False)
        #     pose_errors += pose_error
        
        match_error = 0
        pos_error = 0
        done = False

        if not use_match_gt and match_pred != match_gt:
            match_error == 1

        if obj_id in self.target_obj_ids:
            current_object_pose = dict()
            current_object_pose['t'] = self.get_true_object_pose(obj_id)[:3, 3]
            target_object_pose = self.target_obj_poses[obj_id]['pose']
            pos_error = np.linalg.norm(np.array(current_object_pose['t'])-np.array(target_object_pose['t']))
            # In baselines, they consider an episode to be complete when all target objects are placed within 5 cm from their goal position 
            if pos_error < 0.05: # pose_errors < 0.1 * len(self.target_obj_ids)
                done = True
        else:
            pos, _, _ = self.obj_info(obj_id)
            if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                done = True

        # for obj_id in self.target_obj_ids: # Note that here target object inds maybe different from the initialization, those out of workspace have been removed.
        #     if obj_id not in match_preds.keys():
        #         match_error += 1
        #     else:
        #         obj_ind = self.target_obj_ids.index(obj_id)
        #         if match_preds[obj_id] != obj_ind:
        #             match_error += 1

        #     current_object_pose = dict()
        #     current_object_pose['t'] = self.get_true_object_pose(obj_id)[:3, 3]
        #     target_object_pose = self.target_obj_poses[obj_id]['pose']
        #     pos_error = np.linalg.norm(np.array(current_object_pose['t'])-np.array(target_object_pose['t']))
        #     pos_errors += pos_error

        # # out_of_workspace = []
        # for obj_id in range(len(self.target_obj_ids), len(self.obj_ids["rigid"])):
        #     if obj_id not in match_preds.keys():
        #         match_error += 1
        #     else:
        #         obj_ind = self.target_obj_ids.index(obj_id)
        #         if match_preds[obj_id] != None:
        #             match_error += 1

        #     pos, _, _ = self.obj_info(obj_id)
        #     if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
        #         or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
        #         out_of_workspace.append(obj_id)
        # nontarget_in_workspace = (len(self.target_obj_ids) - len(self.obj_ids["rigid"])) - len(out_of_workspace)
        
        
        print(f"Matching error: {match_error}, position errors: {pos_error}.")
        reward = done - match_error - pos_error

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()

        return reward, done

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        pb.disconnect()

    def get_link_pose(self, body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]

    def add_target_objects(self, num_obj, workspace_limits, save_dir=None):
        """Randomly dropped objects to the workspace"""
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")
        obj_mesh_ind = random.sample(list(range(0, len(mesh_list))), num_obj)
        # obj_mesh_color = color_space[np.asarray(range(num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        object_mesh_files = []
        self.target_obj_ids = [4 + int(i) for i in range(num_obj)]
        self.target_obj_poses = dict()
        
        if save_dir is not None:
            file_name = os.path.join(save_dir, "target.txt")
        else:
            file_name = "cases/temp.txt" 

        with open(file_name, "w") as out_file: # "cases/" 
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                # !!! for 3-dof testing cases generation
                # object_orientation = [
                #     0.,
                #     0.,
                #     2 * np.pi * np.random.random_sample(),
                # ]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )
                object_mesh_files.append(curr_mesh_file)

                self.target_obj_poses[body_id] = dict()
                self.target_obj_poses[body_id]['mesh_file'] = os.path.join(curr_mesh_file.split('.')[0], 'textured_simplified.obj')
                self.target_obj_poses[body_id]['symmetric'] = OBJECT_SYMMETRIC_MAP[curr_mesh_file.split('/')[-1].split('.')[0]]

        
        # get objects of all target objects when stable
        for obj_id in self.target_obj_ids:
            target_obj_pose = self.get_true_object_pose(obj_id)
            self.target_obj_poses[obj_id]['pose'] = dict()
            self.target_obj_poses[obj_id]['pose']['mat'] = target_obj_pose[:3, :3]
            self.target_obj_poses[obj_id]['pose']['t'] = target_obj_pose[:3, 3]

        return True, object_mesh_files

    def add_objects_from_mesh_files(self, object_mesh_files, non_target_num_obj, workspace_limits, save_dir=None):
        """Randomly dropped objects of the assigned kind to the workspace"""

        body_ids = []
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")

        # get non-target objects
        for obj_mesh_file in object_mesh_files:
            if obj_mesh_file in mesh_list:
                mesh_list.remove(obj_mesh_file)
        
        obj_mesh_ind = random.sample(list(range(0, len(mesh_list))), non_target_num_obj)
        
        if save_dir is not None:
            file_name = os.path.join(save_dir, "initial.txt")
        else:
            file_name = "cases/temp.txt" 

        
        object_positions = []
        object_orientations = []
        with open(file_name, "w") as out_file:
            # add target objects
            for curr_mesh_file in object_mesh_files:
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_positions.append(object_position)
                # !!! for 3-dof testing cases generation
                # object_orientation = [
                #     0.,
                #     0.,
                #     2 * np.pi * np.random.random_sample(),
                # ]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                object_orientations.append(object_orientation)
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )
            
            # add non-target objects
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_positions.append(object_position)
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                object_orientations.append(object_orientation)
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

        return True, object_positions, object_orientations

    def add_objects_except_mesh_files(self, object_mesh_files, non_target_num_obj, workspace_limits):
        """Randomly dropped objects of the assigned kind to the workspace"""

        body_ids = []
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")

        # get non-target objects
        for obj_mesh_file in object_mesh_files:
            if obj_mesh_file in mesh_list:
                mesh_list.remove(obj_mesh_file)

        obj_mesh_ind = random.sample(list(range(0, len(mesh_list))), non_target_num_obj)

        with open("cases/" + "temp.txt", "w") as out_file:
            # add non-target objects
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

        return True

    def add_object_push_from_file(self, file_name, target_file=False):
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = file_content_curr_object[0]
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )

        if target_file:
            self.target_obj_poses = dict()        
            self.target_obj_ids = [4 + int(i) for i in range(num_obj)]

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

            if target_file:
                self.target_obj_poses[body_id] = dict()
                self.target_obj_poses[body_id]['mesh_file'] = os.path.join(curr_mesh_file.split('.')[0], 'textured_simplified.obj')
                self.target_obj_poses[body_id]['symmetric'] = OBJECT_SYMMETRIC_MAP[curr_mesh_file.split('/')[-1].split('.')[0]]

        # give time to stop
        for _ in range(5):
            pb.stepSimulation()
        
        if target_file:
            # get objects of all target objects when stable
            for obj_id in self.target_obj_ids:
                target_obj_pose = self.get_true_object_pose(obj_id)
                self.target_obj_poses[obj_id]['pose'] = dict()
                self.target_obj_poses[obj_id]['pose']['mat'] = target_obj_pose[:3, :3]
                self.target_obj_poses[obj_id]['pose']['t'] = target_obj_pose[:3, 3]

        return success

    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform

    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms
 
    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            if pos[2] < 0.004: # 0.005
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def move_ee_pose_wo_collision(self, pose, obstacles):
        """Move UR5e to target end effector pose."""
        joints_goal = self.solve_ik(pose)
        joints_path = plan_joint_motion(self.ur5e, self.robot_ik_joints, joints_goal, disabled_collisions=self.robot_collision_links,
                                    obstacles=obstacles)

        if joints_path is None:
            return False
        else:
            for q in joints_path:
                self.move_joints(q)
            return True

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def calculate_pose_offset(self, target_pose, obj_id):
        """Execute placing primitive.

        Args:
            pose: SE(3) object pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        object_current_transform = self.get_true_object_pose(obj_id)
        if target_pose is not None:
            # get link target pose
            object_target_transform = target_pose
            transform_offset = object_target_transform @ np.linalg.inv(object_current_transform)

            translation = transform_offset[:3, 3]
            rotation = R.from_matrix(transform_offset[:3, :3]).as_euler('xyz', degrees=False)
        
        else:
            translation = object_current_transform[:3, 3] - np.array([-0.494, -0.111, object_current_transform[2, 3]])
            rotation = 0
            transform_offset = np.eye(4)
            transform_offset[:3, 3] = translation

        ratio = 0.2
        score = np.linalg.norm(rotation) + ratio * np.linalg.norm(translation)

        return transform_offset, score

    def go_seeing(self, over=None, rot=None):
        if over is None:
            over = np.array([0.5, 0., 0.3])
        if rot is None:
            # rot = np.array([0., 0.70710678, 0., 0.70710678]) 
            rot = np.array([[0., 0.79335334, 0., 0.60876143]])
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = over

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        over = (ee_transform[:3, 3]).T
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()
        # over = np.array([0.45, 0., 0.25])
        # rot = np.array([0., 0., 0., 1])

        success = self.move_ee_pose((over, rot))
        return success

    # move to a pre-defined pose in tip frame
    def move(self, pose):
        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()
        
        success = self.move_joints(self.ik_rest_joints)
        if success:            
            success = self.move_ee_pose((over, rot))        

    # grasp with motion planning
    def grasp(self, pose, speed=0.005):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])
        
        # visualization for debug
        # ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        # tip_pos, tip_rot = self.get_link_pose(self.ee, self.ee_tip_id)
        # ee_axis = DebugAxes()
        # ee_axis.update(ee_pos, ee_rot)
        # tip_axis = DebugAxes()
        # tip_axis.update(tip_pos, tip_rot)

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()
        
        # Execute 6-dof grasping.
        grasped_obj_id = None
        min_pos_dist = None
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        if success:            
            success = self.move_ee_pose((over, rot))
            # success = self.move_ee_pose_wo_collision((over, rot), self.obj_ids["rigid"])
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed/4)
            success &= self.is_gripper_closed
            
            if success: # get grasp object id
                max_height = -0.0001
                for i in self.obj_ids["rigid"]:
                    height = self.info[i][0][2]
                    if height >= max_height:
                        grasped_obj_id = i
                        max_height = height
        
        if not success:                
            self.go_home()

        # if success:
        #     success = self.move_joints(self.drop_joints1)
        #     # success &= self.is_gripper_closed
        #     self.open_gripper(is_slow=True)


        print(f"Grasp at {pose}, the grasp {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success, grasped_obj_id, min_pos_dist

    # place out of the workspace / trash
    def place_out_of_workspace(self):
        """Execute placing primitive.

        Returns:
            success: robot movement success if True.
        """
        success = self.move_joints(self.drop_joints1)
        success &= self.is_gripper_closed
        self.open_gripper(is_slow=True)
        self.go_home() 

        print(f"Move the object to the trash bin, the place {success}")
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )
        return success

    # simply reorient object pose
    def object_reorient(self, obj_id, delta_ori):
        """Interactive Perception.

        Args:
            delta_ori: SE(3) object pose.

        Returns:
            success: robot movement success if True.
        """

        pb.setGravity(0, 0, 0)
        
        object_delta_transform = np.eye(4)
        # eurler to rotation matrix
        # object_delta_transform[:3, :3] = utils.euler2rotm(delta_ori[0])
        object_delta_transform[:3, :3] = R.from_euler('xyz', delta_ori, degrees=False).as_matrix()
        object_delta_transform[3, 2] = 0.2

        # get current object pose
        object_current_transform = self.get_true_object_pose(obj_id)

        object_target_transform = object_current_transform @ object_delta_transform

        pos = (object_target_transform[:3, 3]).T
        rot = R.from_matrix(object_target_transform[:3, :3]).as_quat()

        pb.resetBasePositionAndOrientation(obj_id, pos, rot)

        return True

    # reorient the pose by manipulator
    def reorient(self, delta_ori, initial_over=None, initial_rot=None):
        """Interactive Perception.

        Args:
            delta_ori: SE(3) gripper pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        success = self.go_seeing(over=initial_over, rot=initial_rot)
        for _ in range(15):
            pb.stepSimulation()

        tip_delta_transform = np.eye(4)
        # euler to rotation matrix
        tip_delta_transform[:3, :3] = R.from_euler('xyz', delta_ori, degrees=False).as_matrix()

        tip_pos, tip_rot = self.get_link_pose(self.ee, self.ee_tip_id)
        tip_current_transform = np.eye(4)
        tip_current_transform[:3, :3] = R.from_quat(tip_rot).as_matrix()
        tip_current_transform[:3, 3] = tip_pos
        
        tip_target_transform = tip_current_transform @ tip_delta_transform

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = tip_target_transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        # Execute 6-dof grasping.
        # self.close_gripper()
        success = self.move_ee_pose((pos, rot))
        self.close_gripper()
        success &= self.is_gripper_closed
        
        # if not success:
        #     self.go_home() 

        print(f"Change gripper pose to {pos}, {rot}, the change {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    # TODO pick-n-place with SceneCollisionNet
    def place(self, pose, speed=0.005):
        """Execute placing primitive.

        Args:
            pose: SE(3) placing pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])
        
        # visualization for debug
        # ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        # tip_pos, tip_rot = self.get_link_pose(self.ee, self.ee_tip_id)
        # ee_axis = DebugAxes()
        # ee_axis.update(ee_pos, ee_rot)
        # tip_axis = DebugAxes()
        # tip_axis.update(tip_pos, tip_rot)

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        # Execute 6-dof placing.
        self.close_gripper()
        success = self.move_ee_pose((over, rot))
        # obstacles = self.obj_ids["rigid"]
        # obstacles.remove(grasped_obj_id)
        # success = self.move_ee_pose_wo_collision((over, rot), obstacles)
        if success:
            success = self.straight_move(over, pos, rot, speed/4, detect_force=True)
            self.open_gripper(is_slow=True)
            self.go_home() 

        print(f"Place at {pose}, the place {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    def place_to_buffer(self, speed=0.005):
        """Execute placing primitive.

        Args:
            pose: SE(3) placing pose.

        Returns:
            success: robot movement success if True.
        """

        over, rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        pos = np.array((over[0], over[1], over[2] - 0.2))

        # Execute 6-dof placing.
        success = self.straight_move(over, pos, rot, speed, detect_force=True)
        self.open_gripper(is_slow=True)
        self.go_home() 

        print(f"Place at the buffer, the place {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    def place_delta_pose(self, delta_pose, intermediate=False, speed=0.005):
        """Execute placing primitive.

        Args:
            pose: SE(3) object pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        ee_current_transform = np.eye(4)
        ee_current_transform[:3, :3] = R.from_quat(ee_rot).as_matrix()
        ee_current_transform[:3, 3] = ee_pos
        ee_target_transform = delta_pose @ ee_current_transform

        pos = (ee_target_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_target_transform[:3, :3]).as_quat()

        # Execute 6-dof placing.
        self.close_gripper()
        success = self.move_ee_pose((over, rot))
        # obstacles = self.obj_ids["rigid"]
        # obstacles.remove(grasped_obj_id)
        # success = self.move_ee_pose_wo_collision((over, rot), obstacles)
        if not intermediate:
            if success:
                success = self.straight_move(over, pos, rot, speed, detect_force=True)
                self.open_gripper(is_slow=True)
                self.go_home() 
            print(f"Place at {delta_pose} with predicted offset, the place {success}")
        else:
            print("Reached intermediate pose")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    def place_pose_offset(self, pose, grasped_obj_id, speed=0.005):
        """Execute placing primitive.

        Args:
            pose: SE(3) object pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        # get link target pose
        object_target_transform = pose
        object_current_transform = self.get_true_object_pose(grasped_obj_id)
        transform_offset = object_target_transform @ np.linalg.inv(object_current_transform)
        ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        ee_current_transform = np.eye(4)
        ee_current_transform[:3, :3] = R.from_quat(ee_rot).as_matrix()
        ee_current_transform[:3, 3] = ee_pos
        ee_target_transform = transform_offset @ ee_current_transform

        pos = (ee_target_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_target_transform[:3, :3]).as_quat()

        # Execute 6-dof placing.
        self.close_gripper()
        success = self.move_ee_pose((over, rot))
        # obstacles = self.obj_ids["rigid"]
        # obstacles.remove(grasped_obj_id)
        # success = self.move_ee_pose_wo_collision((over, rot), obstacles)
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
            self.open_gripper(is_slow=True)
            self.go_home() 

        print(f"Place at {pose} with gt offset, the place {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success

    def place_object_pose(self, pose, grasped_obj_id):
        """Place with Object Reorientation.

        Args:
            delta_ori: SE(3) object pose.

        Returns:
            success: object movement success if True.
        """

        object_target_transform = pose

        pos = (object_target_transform[:3, 3]).T
        rot = R.from_matrix(object_target_transform[:3, :3]).as_quat()

        pb.resetBasePositionAndOrientation(grasped_obj_id, pos, rot)

        self.open_gripper(is_slow=True)
        self.go_home() 
        
        return True        

    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()

class DebugAxes(object):
    """
    Visualization of local frame: red for x axis, green for y axis, blue for z axis
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Args:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = pb.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = pb.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = pb.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])

if __name__ == "__main__":
    env = Environment()
    env.reset()

    print(pb.getPhysicsEngineParameters(env._client_id))

    time.sleep(1)
    # env.add_object_push_from_file("hard-cases/temp.txt", switch=None)

    # push_start = [4.280000000000000471e-01, -3.400000000000000244e-02, 0.01]
    # push_end = [5.020000000000000018e-01, -3.400000000000000244e-02, 0.01]
    # env.push(push_start, push_end)
    # time.sleep(1)

    env.render_camera(env.oracle_cams[0])

    for i in range(16):
        best_rotation_angle = np.deg2rad(90 - i * (360.0 / 16))
        primitive_position = [0.6, 0, 0.01]
        primitive_position_end = [
            primitive_position[0] + 0.1 * np.cos(best_rotation_angle),
            primitive_position[1] + 0.1 * np.sin(best_rotation_angle),
            0.01,
        ]
        env.push(primitive_position, primitive_position_end, speed=0.0002)
        env._pb.addUserDebugLine(primitive_position, primitive_position_end, lifeTime=0)

        # angle = np.deg2rad(i * 360 / 16)
        # pos = [0.5, 0, 0.05]
        # env.grasp(pos, angle)

        time.sleep(1)
