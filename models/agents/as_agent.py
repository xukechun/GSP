import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models.networks import ASBlock, QNetwork_, GaussianPolicy


class ASNet(object):
    def __init__(self, args):

        self.device = args.device

        # delat flow feature
        self.feature = ASBlock(args).to(device=self.device)

        # critic
        self.critic = QNetwork_(2048, args.action_dim, args.as_hidden_size).to(device=self.device)            
        self.critic_target = QNetwork_(2048, args.action_dim, args.as_hidden_size).to(self.device)

        # policy
        # self.action_space = np.asarray([[-3.14, -3.14, -3.14], [3.14, 3.14, 3.14]]) # pose space for action scaling
        self.action_space = np.asarray([[-1.57, -1.57, -1.57], [1.57, 1.57, 1.57]])
        # self.action_space = np.asarray([[-0.57, -0.57, -0.57], [0.57, 0.57, 0.57]])
        # self.action_space = np.asarray([[-1., -1., -1.], [1., 1., 1.]])
        self.policy = GaussianPolicy(2048, args.action_dim, args.as_hidden_size, self.action_space)
        self.policy.to(self.device)
        

        if not args.evaluate_as:
            self.feature_optim = Adam(self.feature.parameters(), lr=args.lr)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.match_threshold = torch.tensor(0., requires_grad=True, device=self.device)
            self.match_threshold_optim = Adam([self.match_threshold], lr=args.lr)

            # SAC parameters
            self.gamma = args.gamma
            self.tau = args.tau
            self.alpha = args.alpha # self.alpha = 0
            self.target_update_interval = args.target_update_interval
            self.automatic_entropy_tuning = args.automatic_entropy_tuning

            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(args.action_dim).to(self.device)).item()
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            # hard update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            
            for k,v in self.feature.named_parameters():
                if "clip" in k:
                    v.requires_grad = False # fix parameters
                # print(v.requires_grad)

            self.feature.train()
            self.critic.train()
            self.critic_target.train()
            self.policy.train()

        else:
            self.feature.eval()
            self.critic.eval()
            self.critic_target.eval()
            self.policy.eval()


    def select_action(self, delta_flow, evaluate=False):
        flow_feat = self.feature(delta_flow) # [1, 2048]
        if evaluate is False:
            action, _, _ = self.policy.sample(flow_feat) # [1, 3]
        else:
            _, _, action = self.policy.sample(flow_feat) # [1, 3] mean

        return action.detach().cpu().numpy()


    def forward(self, delta_flow):
        flow_feat = self.feature(delta_flow) # [1, 2048]
        action, _, _ = self.policy.sample(flow_feat) # [1, 3]
        
        qf1, qf2 = self.critic(flow_feat, action) # [1, 2048 + 3]
        return action, qf1, qf2
        

    def update_parameters(self, memory, batch_size, updates):

        # Sample a batch from memory
        is_target_batch, delta_flow_batch, delta_ori_batch, reward_batch, success_batch, next_delta_flow_batch, next_entropy_batch, mask_batch = memory.sample(batch_size=batch_size)

        is_target_batch = torch.FloatTensor(is_target_batch).to(self.device)
        delta_flow_batch = torch.FloatTensor(delta_flow_batch).to(self.device)
        delta_ori_batch = torch.FloatTensor(delta_ori_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        success_batch = torch.FloatTensor(success_batch).to(self.device)
        next_delta_flow_batch = torch.FloatTensor(next_delta_flow_batch).to(self.device)
        next_entropy_batch = torch.FloatTensor(next_entropy_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            next_flow_feat = self.feature(next_delta_flow_batch)
            next_delta_ori, next_state_log_pi, _ = self.policy.sample(next_flow_feat)
            qf1_next_target, qf2_next_target = self.critic_target(next_flow_feat, next_delta_ori)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        flow_feat = self.feature(delta_flow_batch)
        qf1, qf2 = self.critic(flow_feat, delta_ori_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()


        flow_feat = self.feature(delta_flow_batch)
        pi, log_pi, _ = self.policy.sample(flow_feat)

        qf1_pi, qf2_pi = self.critic(flow_feat, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            # soft update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # update matching threshold
        if is_target_batch.item():
            threshold_loss = -np.power(-1, success_batch.item()) * F.leaky_relu(next_entropy_batch - self.match_threshold)
        else:
            threshold_loss = F.relu(next_entropy_batch - self.match_threshold)

        self.match_threshold_optim.zero_grad()
        threshold_loss.backward()
        self.match_threshold_optim.step()
        threshold_tlogs = self.match_threshold.clone()
        
        
        with torch.no_grad():
            next_flow_feat = self.feature(next_delta_flow_batch)
            next_delta_ori, next_state_log_pi, _ = self.policy.sample(next_flow_feat)
            qf1_next_target, qf2_next_target = self.critic_target(next_flow_feat, next_delta_ori)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        flow_feat = self.feature(delta_flow_batch)
        qf1, qf2 = self.critic(flow_feat, delta_ori_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss_ = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss_ = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(flow_feat)

        qf1_pi, qf2_pi = self.critic(flow_feat, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss_ = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        total_loss = 0.5 * (0.5 * qf1_loss_ + 0.5 * qf2_loss_) + 0.5 * policy_loss_

        self.feature_optim.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.vig_fusion.parameters(), 0.1)
        self.feature_optim.step()
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), total_loss.item(), threshold_tlogs.item(), threshold_loss.item()
