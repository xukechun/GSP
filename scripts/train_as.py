import os
import time
import argparse
import numpy as np
import random
import datetime
import torch
from tensorboardX import SummaryWriter

import helpers.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from helpers.logger import Logger
from helpers.grasp_detetor import Graspnet
from helpers.flow_detector import OpticalFlowNet
from helpers.replay_memory import ASReplayMemory
from models.as_agent import ASNet
from models.networks import CustomCLIP


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--evaluate_as', action='store', type=bool, default=False)
    parser.add_argument('--load_model', action='store', type=bool, default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')
    parser.add_argument('--clip_model_path', action='store', type=str, default='')
    parser.add_argument('--save_model_interval', type=int, default=500, metavar='N',
                        help='episode interval to save model')

    parser.add_argument('--num_obj', action='store', type=int, default=7)
    parser.add_argument('--non_target_num_obj', action='store', type=int, default=0)
    parser.add_argument('--num_scene', action='store', type=int, default=4000)
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
    parser.add_argument('--max_episode_step', type=int, default=5)
    parser.add_argument('--dense_reward', action='store', type=bool, default=True)
    parser.add_argument('--visualize', action='store', type=bool, default=False)

    # RAFT paras
    parser.add_argument('--model', help="restore checkpoint", default='models/RAFT/models/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        
    # Matcher paras
    parser.add_argument('--matcher', action='store', type=str, default='CLIP')
    parser.add_argument('--patch_size', type=int, default=32)

    # SAC paras
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--as_hidden_size', type=int, default=1024, metavar='N',
                        help='hidden size (default: 1024)')
    parser.add_argument('--action_dim', type=int, default=3, metavar='N',
                        help='action dim (default: 3)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=512, metavar='N',
                        help='size of replay buffer (default: 512)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    
    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_obj = args.num_obj
    num_scene = args.num_scene

    # load environment
    env = Environment(gui=False)
    env.seed(args.seed)
    # load logger
    logger = Logger(suffix="as")
    # load graspnet
    graspnet = Graspnet()
    # load flownet
    flownet = OpticalFlowNet(args)
    # load custom clip
    customclip = CustomCLIP(args)
    customclip.load_state_dict(torch.load(args.clip_model_path))
    # load active seeing policy
    agent = ASNet(args)

    if args.load_model:
        logger.load_as_checkpoint(agent, args.model_path, args.evaluate_as)

    #Tesnorboard
    writer = SummaryWriter("tensor_logs/{}_as_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                            "autotune" if args.automatic_entropy_tuning else "fixed"))
    # Memory
    memory = ASReplayMemory(args.replay_size, args.seed)

    # training
    iteration = 0
    updates = 0
    episode = 0
    
    for scene in range(num_scene):

        reset = False
        # generate target configuration
        while not reset:
            env.reset()
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

            # record target information
            logger.save_target_heightmaps(iteration, target_color_image, target_depth_image)
            logger.save_target_bbox_images(iteration, target_remain_bbox_images)

        if len(target_bbox_padding_images) != len(target_remain_bbox_images) or len(target_bbox_padding_images) != len(target_object_mesh_files):
            print("Wrong target detection!!! Reset the environment!!")
            continue

        # generate initial configuration
        for i in range(len(target_object_mesh_files) + args.non_target_num_obj):
            episode_reward = 0
            episode_steps = 0
            done = False
            reset = False
            success = False
            episilo = min(0.6 * np.power(1.0002, episode), 0.99)

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

            # active seeing for matching
            while not done:  
                punish = []   
                if episode_steps == 0:
                    # grasp the object
                    pcd = utils.get_fuse_pointcloud(env)
                    with torch.no_grad():
                        grasp_pose_set, _, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses())
                    print("Number of grasping poses", len(grasp_pose_set))
                    
                    if len(grasp_pose_set) == 0:
                        print("No grasp for this object!")
                        break

                    action_idx = np.random.randint(0, len(grasp_pose_set))
                    action = grasp_pose_set[action_idx]
                    grasp_success, _, _ = env.grasp(action)

                    grasp_failure_num = 0
                    if not grasp_success:
                        grasp_failure_num += 1
                        if grasp_failure_num >= 7:
                            break
                        else:
                            continue
                    
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

                    # flow to matched target object
                    # prob, match, score, entropy = agent.feature.get_clip_match_dist(obj_bboxes, target_bboxes)
                    prob, match, score, entropy = customclip.get_customclip_match_dist(obj_bboxes, target_bboxes)
                    # utils.plot_attnmap(prob[0].detach().cpu().numpy())
                        
                    match_eval = match.item()

                    if match_eval == match_gt:
                        done = True
                        success = True
                        episode += 1
                        break                                    

                    single_flow = flownet.run(obj_bbox_padding_image[0], target_bbox_padding_images[match_eval], vis=args.visualize) # Note that we assume the target bbox is true 
                    # obj_flow = flow_matched[:, :, bbox_pixel[0]:bbox_pixel[1], bbox_pixel[2]:bbox_pixel[3]]
                    # flow to the whole target image
                    global_flow = flownet.run(obj_bbox_padding_image[0], target_color_image, vis=args.visualize)
                    # flow uncertainty
                    delta_flow = global_flow - single_flow

                # visualization
                if args.visualize:
                    flownet.viz_uncertainty(obj_bbox_padding_image[0], delta_flow)
                if np.random.randn() <= episilo: # greedy policy
                    with torch.no_grad():
                        # delta euler pose
                        delta_ori = agent.select_action(delta_flow)
                else:
                    # delta_ori = np.random.rand(1, 3) * 6.28 - 3.14
                    delta_ori = np.random.rand(1, 3) * 3.14 - 1.57
                    # success = env.object_reorient(4, delta_ori)

                if len(memory) >= args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, feature_loss, threshold, threshold_loss = agent.update_parameters(memory, args.batch_size, updates)
                        # loss = critic_1_loss + critic_2_loss + policy_loss
                        # writer.add_scalar('loss/loss', loss, updates)
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('loss/feature', feature_loss, updates)
                        writer.add_scalar('loss/threshold', threshold_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        writer.add_scalar('threshold', threshold, updates)
                        updates += 1

                reorient_success = env.reorient(delta_ori)
                punish.append(0.2 * (1 - reorient_success) + 0.18 * np.linalg.norm(delta_ori))

                # next state, state after active seeing
                next_color_image, next_depth_image, next_mask_image = utils.get_true_heightmap(env)
                next_obj_bbox_images, next_obj_bbox_positions = utils.get_true_bboxs(env, next_color_image, next_depth_image, next_mask_image)
                next_obj_bbox_padding_image, _ = utils.get_true_bbox_padding_images(env, next_color_image, next_mask_image)
                if len(next_obj_bbox_padding_image)==0:
                    done = True
                    break

                _, next_obj_resized_bbox_images, next_obj_bboxes, _ = utils.bbox_preprocess(next_obj_bbox_images, next_obj_bbox_positions, (args.patch_size, args.patch_size))


                if next_obj_bboxes == None:
                    done = True
                    break

                # flow to matched target object
                # next_prob, next_match, next_score, next_entropy = agent.feature.get_clip_match_dist(next_obj_bboxes, target_bboxes)
                next_prob, next_match, next_score, next_entropy = customclip.get_customclip_match_dist(next_obj_bboxes, target_bboxes)

                next_match_eval = next_match.item()
                next_single_flow = flownet.run(next_obj_bbox_padding_image[0], target_bbox_padding_images[next_match_eval]) # Note that we assume the target bbox is true 
                # obj_flow = flow_matched[:, :, bbox_pixel[0]:bbox_pixel[1], bbox_pixel[2]:bbox_pixel[3]]
                # flow to the whole target image
                next_global_flow = flownet.run(next_obj_bbox_padding_image[0], target_color_image)
                # flow uncertainty
                next_delta_flow = next_global_flow - next_single_flow

                if is_target:
                    last_success = match_eval == match_gt
                    success = next_match_eval == match_gt
                    if success: 
                        done = True
                    
                    if args.dense_reward:
                        reward = (success + entropy - next_entropy).detach().cpu().numpy().item() - sum(punish) / len(punish)
                    else:
                        reward = success

                else:
                    last_done = False
                    done = False
                    # done = episode_steps == args.max_episode_step
                    reward = 0

                episode_steps += 1
                iteration += 1
                episode_reward += reward
                print("\033[034m Episode: {}, total numsteps: {}, reward: {}\033[0m".format(episode, iteration, round(reward, 2), done))
                
                # Ignore the "done" signal if it comes from hitting the max step horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == args.max_episode_step else float(not done)

                memory.push(is_target, delta_flow.detach().cpu().numpy(), delta_ori, reward, done, next_delta_flow.detach().cpu().numpy(), next_entropy.detach().cpu().numpy()[0], mask) # Append transition to memory
                
                # record
                logger.save_heightmaps(iteration, color_image, depth_image)
                logger.save_bbox_images(iteration, obj_bbox_images)
                logger.reward_logs.append(reward)
                logger.executed_action_logs.append(delta_ori)
                logger.write_to_log('reward', logger.reward_logs)
                # logger.write_to_log('executed_action', logger.executed_action_logs)
                writer.add_scalar('reward/step', reward, iteration)
                
                if episode_steps == args.max_episode_step:
                    done = True
                
                if done:
                    episode += 1
                    break

                delta_flow = next_delta_flow
                entropy = next_entropy

            if episode % args.save_model_interval == 0:
                logger.save_as_checkpoint(agent, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), str(episode))

            logger.episode_reward_logs.append(episode_reward)
            logger.episode_step_logs.append(episode_steps)
            logger.episode_success_logs.append(success)
            writer.add_scalar('reward/episode', episode_reward, episode)
            writer.add_scalar('step', episode_steps, episode)
            writer.add_scalar('success', success, episode)
            logger.write_to_log('episode_reward', logger.episode_reward_logs)
            logger.write_to_log('episode_step', logger.episode_step_logs)
            logger.write_to_log('episode_success', logger.episode_success_logs)
            print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode-1, iteration, episode_steps, round(episode_reward, 2), success))