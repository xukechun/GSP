import os
import cv2
import argparse
import numpy as np
import random
import torch

import helpers.utils as utils
from env.environment_sim import Environment
from helpers.logger import Logger
from helpers.grasp_detetor import Graspnet
from helpers.flow_detector import OpticalFlowNet
from models.agents.as_agent import ASNet 
from models.networks import CustomCLIP

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--use_thres', action='store', type=bool, default=False)
    parser.add_argument('--ent_thres', type=float, default=0.18)
    parser.add_argument('--score_thres', type=float, default=0.85, metavar='N',
                    help='score threshold for confident matching (default: 0.95)') 
    parser.add_argument('--evaluate_as', action='store', type=bool, default=True)
    parser.add_argument('--testing_cases_dir', action='store', type=str, default='testing_cases/')
    parser.add_argument('--bad_testing_cases_dir', action='store', type=str, default='testing_cases/as_cases/bad_cases')
    parser.add_argument('--load_model', action='store', type=bool, default=False)
    parser.add_argument('--model_path', action='store', type=str, default='')
    # parser.add_argument('--clip_model_path', action='store', type=str, default='supervised_logs/clip_models/clip_supervised_2023-03-06.15:10:46.pth')
    parser.add_argument('--clip_model_path', action='store', type=str, default='supervised_logs/clip_models/clip_supervised_399_2023-03-09.10:49:08.pth')
    # parser.add_argument('--clip_model_path', action='store', type=str, default='supervised_logs/clip_models/clip_supervised_399_2023-03-22.22:03:46.pth')
    parser.add_argument('--max_episode_step', type=int, default=5)
    parser.add_argument('--visualize', action='store', type=bool, default=False)
    parser.add_argument('--suffix', action='store', type=str, default='as')

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

    # load environment
    env = Environment(gui=True)
    env.seed(args.seed)
    # load logger
    logger = Logger(suffix=args.suffix, case_dir=args.testing_cases_dir)
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

    case_dirs = os.listdir(args.testing_cases_dir)
    
    bad_case_dirs = os.listdir(args.bad_testing_cases_dir)
    
    single_success_num = 0
    failed_grasp_number = 0
    failed_detection_number = 0
    bad_detection_number = 0
    failed_match_number = 0
    wrong_small_ent = 0
    one_step_scores = []
    avg_success = []
    avg_steps = []
    avg_delta_norm = []
    avg_reorient_success = []
    bad_reorient = 0
    
    for i in range(len(case_dirs)):
        case_dir = case_dirs[i]
        if case_dir in bad_case_dirs:
            continue
        print(case_dir)
        match_ids = []
        episode_reward = 0
        episode_steps = 0
        iteration = 0
        done = False
        reset = False
        success = False

        files = os.listdir(os.path.join(args.testing_cases_dir, case_dir))
        # add objects in workspapce
        for f in files:
            if f.split(".")[-1] == "txt":
                while not reset:
                    env.reset()
                    reset = env.add_object_push_from_file(os.path.join(args.testing_cases_dir, case_dir, f))
                match_gt = int(f.split(".")[0].split("_")[-1])
                files.remove(f)
        
        # load target bbox images and preprocess
        files.sort(key=lambda x: x.split('.')[0])
        target_bbox_images = []
        target_bbox_padding_images = []
        for f in files:
            bbox_image = cv2.imread(os.path.join(args.testing_cases_dir, case_dir, f))
            bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            if f.split(".")[-2] == "bbox":
                target_bbox_images.append(bbox_image)
            elif f.split(".")[-2] == "bbox_padding":
                target_bbox_padding_images.append(bbox_image)
            elif f.split(".")[-2] == "color":
                target_color_image = bbox_image
                
        target_obj_num = len(target_bbox_images)
        print(f"\033[032m Case {i}, there are {target_obj_num} target objects! \033[0m")
               
        if args.matcher == "CLIP":
            target_bboxes = utils.target_bbox_preprocess(target_bbox_images, (args.patch_size, args.patch_size))

        # active seeing for matching
        while not done:     
            punish = []
            if episode_steps == 0:
                # grasp the object
                pcd = utils.get_fuse_pointcloud(env)

                with torch.no_grad():
                    # best_grasp_pose, _ = graspnet.single_grasp_detection(pcd, visualize=args.visualize)
                    best_grasp_pose, _ = graspnet.single_grasp_detection(pcd, env.get_true_object_poses(), visualize=args.visualize)
                    
                if best_grasp_pose is None:
                    failed_grasp_number += 1
                    print("No grasp for this object!")
                    break

                action = best_grasp_pose
                grasp_success, _, _ = env.grasp(action)

                if not grasp_success:
                    continue
                
                # render front view
                color_front, _, _ = env.render_camera(env.agent_cams[0])
                color_topdown, _, _ = env.render_camera(env.oracle_cams[0]) 
                color_image, depth_image, mask_image = utils.get_true_heightmap(env)
                obj_bbox_images, obj_bbox_positions = utils.get_true_bboxs(env, color_image, depth_image, mask_image)
                obj_bbox_padding_image, _ = utils.get_true_bbox_padding_images(env, color_image, mask_image)

                if len(obj_bbox_padding_image)==0:
                    failed_detection_number += 1
                    print("No detection for this object before active seeing!")
                    break

                if args.matcher == "CLIP":
                    _, obj_resized_bbox_images, obj_bboxes, obj_pos_bboxes = utils.bbox_preprocess(obj_bbox_images, obj_bbox_positions, (args.patch_size, args.patch_size))

                if obj_bboxes == None:
                    failed_detection_number += 1
                    print("No detection for this object before active seeing!")
                    break
                
                logger.save_front_images(iteration, episode_steps, color_front)
                logger.save_topdown_images(iteration, episode_steps, color_topdown)
                logger.save_heightmaps(iteration, color_image, depth_image)
                logger.save_bbox_images(iteration, obj_bbox_images)
                
                # flow to matched target object
                if args.matcher == "CLIP":
                    prob, match, score, entropy = customclip.get_customclip_match_dist(obj_bboxes, target_bboxes)
                    # prob, match, score, entropy = agent.feature.get_clip_match_dist(obj_bboxes, target_bboxes)
                    match_ids.append(match.item())
                    print('before CLIP score: ', prob.detach().cpu().numpy()[0])
                    print('entropy before CLIP matching: ', entropy.item())


                match_eval = match.item()

                if not args.use_thres:
                    if match_eval == match_gt:
                        done = True
                        success = True
                        single_success_num += 1
                        break  
                else:
                    if args.matcher == "CLIP":
                        args.score_thres = 1./target_bboxes.shape[1] + 0.12
                    # if entropy.item() <= args.ent_thres or score.item() > args.score_thres:
                    if score.item() > args.score_thres:    
                        one_step_scores.append(score.item())
                        done = True
                        success = match_eval == match_gt
                        if not success:
                            wrong_small_ent += 1
                            print("Wrong high confidence!")
                        break

                single_flow = flownet.run(obj_bbox_padding_image[0], target_bbox_padding_images[match_eval]) # Note that we assume the target bbox is true
                
                # flow to the whole target image
                global_flow = flownet.run(obj_bbox_padding_image[0], target_color_image)
                
                if args.visualize:
                    flownet.viz_magnitude(single_flow)
                    flownet.viz_magnitude(global_flow)
                    
                # flow uncertainty
                delta_flow = global_flow - single_flow

            # visualization
            if args.visualize:
                flownet.viz_uncertainty(obj_bbox_padding_image[0], delta_flow)
                flownet.viz_magnitude(delta_flow)
                
            with torch.no_grad():
                # delta euler pose
                delta_ori = agent.select_action(delta_flow, evaluate=True)
            # if episode_steps == 0:
            #     success = env.go_seeing()
            reorient_success = env.reorient(delta_ori)
            avg_delta_norm.append(np.linalg.norm(delta_ori))
            avg_reorient_success.append(reorient_success)
            punish.append(0.2 * (1 - reorient_success) + 0.18 * np.linalg.norm(delta_ori))

            # next state, state after active seeing
            next_color_front, _, _ = env.render_camera(env.agent_cams[0])
            next_color_topdown, _, _ = env.render_camera(env.oracle_cams[0]) 
            next_color_image, next_depth_image, next_mask_image = utils.get_true_heightmap(env)
            next_obj_bbox_images, next_obj_bbox_positions = utils.get_true_bboxs(env, next_color_image, next_depth_image, next_mask_image)
            next_obj_bbox_padding_image, _ = utils.get_true_bbox_padding_images(env, next_color_image, next_mask_image)

            if len(next_obj_bbox_padding_image)==0:
                # most_match = max(match_ids, key=match_ids.count)
                # success = most_match == match_gt
                logger.save_front_images(iteration, episode_steps+1, next_color_front)
                logger.save_topdown_images(iteration, episode_steps+1, next_color_topdown)
                logger.save_heightmaps(iteration, next_color_image, next_depth_image)
                logger.save_bbox_images(iteration, next_obj_bbox_images)
                print("Bad detection for this object after active seeing!")
                bad_detection_number += 1
                break

            if args.matcher == "CLIP":
                _, next_obj_resized_bbox_images, next_obj_bboxes, _ = utils.bbox_preprocess(next_obj_bbox_images, next_obj_bbox_positions, (args.patch_size, args.patch_size))

            if next_obj_bboxes == None:
                # most_match = max(match_ids, key=match_ids.count)
                # success = most_match == match_gt
                logger.save_front_images(iteration, episode_steps+1, next_color_front)
                logger.save_topdown_images(iteration, episode_steps+1, next_color_topdown)
                logger.save_heightmaps(iteration, next_color_image, next_depth_image)
                logger.save_bbox_images(iteration, next_obj_bbox_images)
                print("Bad detection for this object after active seeing!")
                bad_detection_number += 1
                break

            # flow to matched target object
            if args.matcher == "CLIP":
                next_prob, next_match, next_score, next_entropy = customclip.get_customclip_match_dist(next_obj_bboxes, target_bboxes)
                # next_prob, next_match, next_score, next_entropy = agent.feature.get_clip_match_dist(next_obj_bboxes, target_bboxes)
                match_ids.append(next_match.item())
                print('after CLIP score: ', next_prob.detach().cpu().numpy()[0])
                print('entropy after CLIP matching: ', next_entropy.item())


            next_match_eval = next_match.item()
            next_single_flow = flownet.run(next_obj_bbox_padding_image[0], target_bbox_padding_images[next_match_eval]) # Note that we assume the target bbox is true
            if args.visualize:
                flownet.viz_magnitude(next_single_flow)
            
            # flow to the whole target image
            next_global_flow = flownet.run(next_obj_bbox_padding_image[0], target_color_image)
            if args.visualize:
                flownet.viz_magnitude(next_global_flow) 

            # flow uncertainty
            next_delta_flow = next_global_flow - next_single_flow
            if args.visualize:
                flownet.viz_uncertainty(next_obj_bbox_padding_image[0], next_delta_flow)
                flownet.viz_magnitude(next_delta_flow) 

            success = next_match_eval == match_gt
            reward = (success + entropy - next_entropy).detach().cpu().numpy().item() - sum(punish) / len(punish)
            episode_steps += 1
            iteration += 1
            episode_reward += reward

            # record
            logger.save_front_images(iteration, episode_steps, next_color_front)
            logger.save_topdown_images(iteration, episode_steps, next_color_topdown)
            logger.save_heightmaps(iteration, next_color_image, next_depth_image)
            logger.save_bbox_images(iteration, next_obj_bbox_images)
            logger.reward_logs.append(reward)
            logger.write_to_log('reward', logger.reward_logs)

            if not args.use_thres:
                if success or episode_steps == args.max_episode_step:
                    done = True
            else:
                # if next_entropy.item() <= args.ent_thres or episode_steps == args.max_episode_step or next_score.item() > args.score_thres:
                if episode_steps == args.max_episode_step or next_score.item() > args.score_thres:    
                    done = True
                if next_entropy.item() > args.ent_thres and episode_steps == args.max_episode_step:
                    # choose the most matched one
                    most_match = max(match_ids, key=match_ids.count)
                    success = most_match == match_gt
                    if not success:
                        failed_match_number += 1
                    # success = False

            delta_flow = next_delta_flow
            entropy = next_entropy

        avg_success.append(success)
        if success and episode_steps > 0:
            avg_steps.append(episode_steps)

        logger.episode_reward_logs.append(episode_reward)
        logger.episode_step_logs.append(episode_steps)
        logger.episode_success_logs.append(success)
        logger.write_to_log('episode_reward', logger.episode_reward_logs)
        logger.write_to_log('episode_step', logger.episode_step_logs)
        logger.write_to_log('episode_success', logger.episode_success_logs)
        print("\033[034m Case: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(i-1, iteration, episode_steps, round(episode_reward, 2), success))

    
    if not args.use_thres:
        print("\033[034m Avg success: {} ({}, valid sample: {}), avg steps: {}, avg delta norm: {}, avg reorient success: {}, single success number: {}, failed grasp number: {}\033[0m".format(sum(avg_success)/len(avg_success), sum(avg_success)/(len(avg_success)-failed_grasp_number-failed_detection_number-bad_detection_number), \
                        len(avg_success)-failed_grasp_number-failed_detection_number-bad_detection_number, sum(avg_steps)/len(avg_steps), sum(avg_delta_norm)/len(avg_delta_norm), sum(avg_reorient_success)/len(avg_reorient_success), single_success_num, failed_grasp_number))
    else:
        print("\033[034m Avg success: {} ({}, valid sample: {}), avg steps: {}, avg delta norm: {}, avg reorient success: {}, one step number: {}, failed match number: {}, failed grasp number: {}, wrong small ent: {}\033[0m".format(sum(avg_success)/len(avg_success), sum(avg_success)/(len(avg_success)-failed_grasp_number-failed_detection_number-bad_detection_number), \
                        len(avg_success)-failed_grasp_number-failed_detection_number-bad_detection_number, sum(avg_steps)/len(avg_steps), sum(avg_delta_norm)/len(avg_delta_norm), sum(avg_reorient_success)/len(avg_reorient_success), len(one_step_scores), failed_match_number, failed_grasp_number, wrong_small_ent))
