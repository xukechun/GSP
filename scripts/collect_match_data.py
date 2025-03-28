import os
import time
import argparse
import numpy as np
import random
import datetime
import torch

import helpers.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=111, metavar='N', 
                    help='random seed (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--evaluate_as', action='store', type=bool, default=False)

    parser.add_argument('--num_obj', action='store', type=int, default=7) 
    parser.add_argument('--non_target_num_obj', action='store', type=int, default=5)
    parser.add_argument('--num_scene', action='store', type=int, default=1000) 

    parser.add_argument('--patch_size', type=int, default=32)

    args = parser.parse_args()
    return args


def collect_match_data(args):
    # parameters
    num_obj = args.num_obj
    num_scene = args.num_scene

    # load environment
    env = Environment(gui=False)
    env.seed(args.seed)
    # collect data
    data = dict()
    data['clip_obj_bbox'] = []
    data['clip_target_bboxes'] = []
    data['match'] = []

    # training
    episode = 0
    
    for scene in range(num_scene):
        reset = False
        # generate target configuration
        while not reset:
            env.reset()
            # to make the target objects sparse, so that target detection is ground-truth, or to make the workspace larger
            if scene < 200:
                warmup_num_obj = 5
                reset, target_object_mesh_files = env.add_target_objects(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                reset, target_object_mesh_files = env.add_target_objects(num_obj, WORKSPACE_LIMITS)
            print(f"\033[032m Generate the target configuration of scene {scene}\033[0m")

            # check if all of the target objects are in the workspace:
            raw_target_object_mesh_files = target_object_mesh_files[:]
            for obj_id in env.target_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    env.remove_target_id(obj_id)
                    target_object_mesh_files.remove(raw_target_object_mesh_files[obj_id-4])

            print("Number of target objects in the workspace:", len(env.target_obj_ids))

            # generate the target images and target bboxes
            # !! Note here we assume the target bbox is true
            target_color_image, target_depth_image, target_mask_image = utils.get_true_heightmap(env)
            target_bbox_images, target_bbox_positions = utils.get_true_bboxs(env, target_color_image, target_depth_image, target_mask_image)
            target_bbox_padding_images, target_bbox_pixels = utils.get_true_bbox_padding_images(env, target_color_image, target_mask_image)
            # target_obj_poses = env.get_true_object_poses()

            # preprocess of target bboxes
            target_remain_bbox_images, target_resized_bbox_images, target_bboxes, target_pos_bboxes = utils.bbox_preprocess(target_bbox_images, target_bbox_positions, (args.patch_size, args.patch_size))
            if target_bboxes == None:
                break

        if len(target_bbox_padding_images) != len(target_remain_bbox_images) or len(target_bbox_padding_images) != len(target_object_mesh_files):
            print("Wrong target detection!!! Reset the environment!!")
            continue

        # generate initial configuration
        for i in range(len(target_object_mesh_files) + args.non_target_num_obj):
            reset = False

            while not reset:
                env.reset()
                # for each scene, match each target object and non-target objects of fixed number
                if i < len(target_object_mesh_files):
                    reset, obj_pos, obj_rot = env.add_objects_from_mesh_files([target_object_mesh_files[i]], 0, WORKSPACE_LIMITS)
                    print(f"\033[032m Episode {episode}: generate the initial configuration of target object\033[0m")
                    is_target = True
                    match_gt = i
                else:
                    reset = env.add_objects_except_mesh_files(target_object_mesh_files, 1, WORKSPACE_LIMITS)
                    print(f"\033[032m Episode {episode}: generate the initial configuration of non-target object\033[0m")
                    is_target = False
                    match_gt = -1
                    
            color_image, depth_image, mask_image = utils.get_true_heightmap(env)
            obj_bbox_images, obj_bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)
            obj_bbox_padding_image, _ = utils.get_true_bbox_padding_images(env, color_image, mask_image)
            if len(obj_bbox_padding_image)==0:
                print("Fail to detect the object!")
                break

            _, obj_resized_bbox_images, obj_bboxes, obj_pos_bboxes = utils.bbox_preprocess(obj_bbox_images, obj_bbox_positions, (args.patch_size, args.patch_size))
            if obj_bboxes == None:
                print("Fail to detect the object!")
                break
        

            data['clip_obj_bbox'].append(obj_bboxes.detach().cpu().numpy())
            data['clip_target_bboxes'].append(target_bboxes.detach().cpu().numpy())
            data['match'].append(match_gt)

            if len(data['clip_obj_bbox']) % 3000 == 0:
                timestamp = time.time()
                timestamp_value = datetime.datetime.fromtimestamp(timestamp)
                name = 'match_new_' + timestamp_value.strftime('%Y_%m_%d_%H_%M_%S_') + str(len(data['clip_obj_bbox'])) + '.npy'
                save_path = os.path.join(os.path.dirname(__file__), 'data', name)
                np.save(save_path, data)


if __name__ == "__main__":

    args = parse_args()
    
    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    collect_match_data(args=args)
