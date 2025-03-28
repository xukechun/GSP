import os
import random
import pickle
import numpy as np
from operator import itemgetter

from pyrsistent import immutable

class ASReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0

        self.is_target_buffer = []
        self.delta_flow_buffer = []
        self.delta_ori_buffer = []
        self.reward_buffer = []
        self.success_buffer = []
        self.done_buffer = []
        self.next_delta_flow_buffer = []
        self.next_entropy_buffer = []


    def push(self, is_target, delta_flow, delta_ori, reward, success, next_delta_flow, next_entropy, done):
        if len(self.delta_flow_buffer) < self.capacity:
            self.is_target_buffer.append(None)
            self.delta_flow_buffer.append(None)
            self.delta_ori_buffer.append(None)
            self.reward_buffer.append(None)
            self.success_buffer.append(None)
            self.done_buffer.append(None)
            self.next_delta_flow_buffer.append(None)
            self.next_entropy_buffer.append(None)
        
        # !!! newaxis for batch size 1 !!!
        self.is_target_buffer[self.position] = np.array([is_target])[np.newaxis, :]
        self.delta_flow_buffer[self.position] = delta_flow
        self.delta_ori_buffer[self.position] = delta_ori
        self.reward_buffer[self.position] = np.array([reward])[np.newaxis, :]
        self.success_buffer[self.position] = np.array([success])[np.newaxis, :]
        self.next_delta_flow_buffer[self.position] = next_delta_flow
        self.next_entropy_buffer[self.position] = next_entropy
        self.done_buffer[self.position] = np.array([done])[np.newaxis, :]

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(range(len(self.delta_flow_buffer)), batch_size) 
        
        is_target_batch = itemgetter(*batch)(self.is_target_buffer)
        delta_flow_batch = itemgetter(*batch)(self.delta_flow_buffer)
        delta_ori_batch = itemgetter(*batch)(self.delta_ori_buffer)
        reward_batch = itemgetter(*batch)(self.reward_buffer)
        success_batch = itemgetter(*batch)(self.success_buffer)
        done_batch = itemgetter(*batch)(self.done_buffer)
        next_delta_flow_batch = itemgetter(*batch)(self.next_delta_flow_buffer)
        next_entropy_batch = itemgetter(*batch)(self.next_entropy_buffer)  
        
        return is_target_batch, delta_flow_batch, delta_ori_batch, reward_batch, success_batch, next_delta_flow_batch, next_entropy_batch, done_batch

    def __len__(self):
        return len(self.delta_flow_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0

        self.target_bboxes_buffer = []
        self.target_pos_bboxes_buffer = []
        self.bboxes_buffer = []
        self.pos_bboxes_buffer = []
        self.grasps_buffer = []
        self.next_bboxes_buffer = []
        self.next_pos_bboxes_buffer = []
        self.next_grasps_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []


    def push(self, target_bbox, target_pos_bbox, bbox, pos_bbox, grasp, action, reward, next_bbox, next_pos_bbox, next_grasp, done):
        if len(self.bboxes_buffer) < self.capacity:
            self.target_bboxes_buffer.append(None)
            self.target_pos_bboxes_buffer.append(None)
            self.bboxes_buffer.append(None)
            self.pos_bboxes_buffer.append(None)
            self.grasps_buffer.append(None)
            self.next_bboxes_buffer.append(None)
            self.next_pos_bboxes_buffer.append(None)
            self.next_grasps_buffer.append(None)
            self.action_buffer.append(None)
            self.reward_buffer.append(None)
            self.done_buffer.append(None)
        
        # !!! newaxis for batch size 1 !!!
        self.target_bboxes_buffer[self.position] = target_bbox[np.newaxis, :]
        self.target_pos_bboxes_buffer[self.position] = target_pos_bbox[np.newaxis, :]
        self.bboxes_buffer[self.position] = bbox[np.newaxis, :]
        self.pos_bboxes_buffer[self.position] = pos_bbox[np.newaxis, :]
        self.grasps_buffer[self.position] = grasp[np.newaxis, :]
        self.next_bboxes_buffer[self.position] = next_bbox[np.newaxis, :]
        self.next_pos_bboxes_buffer[self.position] = next_pos_bbox[np.newaxis, :]
        self.next_grasps_buffer[self.position] = next_grasp[np.newaxis, :]
        self.action_buffer[self.position] = np.array([action])[np.newaxis, :]
        self.reward_buffer[self.position] = np.array([reward])[np.newaxis, :]
        self.done_buffer[self.position] = np.array([done])[np.newaxis, :]

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(range(len(self.bboxes_buffer)), batch_size) 
        
        target_bboxes_batch = itemgetter(*batch)(self.target_bboxes_buffer)
        target_pos_bboxes_batch = itemgetter(*batch)(self.target_pos_bboxes_buffer)
        bboxes_batch = itemgetter(*batch)(self.bboxes_buffer)
        pos_bboxes_batch = itemgetter(*batch)(self.pos_bboxes_buffer)
        grasps_batch = itemgetter(*batch)(self.grasps_buffer)
        action_batch = itemgetter(*batch)(self.action_buffer)
        reward_batch = itemgetter(*batch)(self.reward_buffer)
        done_batch = itemgetter(*batch)(self.done_buffer)
        next_bboxes_batch = itemgetter(*batch)(self.next_bboxes_buffer)
        next_pos_bboxes_batch = itemgetter(*batch)(self.next_pos_bboxes_buffer)
        next_grasps_batch = itemgetter(*batch)(self.next_grasps_buffer) 
        
        return target_bboxes_batch, target_pos_bboxes_batch, bboxes_batch, pos_bboxes_batch, grasps_batch, action_batch, reward_batch, done_batch, next_bboxes_batch, next_pos_bboxes_batch, next_grasps_batch

    def __len__(self):
        return len(self.bboxes_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
