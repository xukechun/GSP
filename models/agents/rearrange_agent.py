import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.optim import Adam
from models.networks import ViGFusion, ViGFusion_Adapter, QNetwork, Policy


class GViR(object):
    def __init__(self, grasp_dim, args):

        self.device = args.device

        # state-action feature
        if args.use_adapter:
            self.vig_fusion = ViGFusion_Adapter(grasp_dim, args.width, args.layers, args.heads, self.device).to(device=self.device)
            # load adapter parameters
            adapter_dict = torch.load(args.clip_model_path)
            adapter_dict = {k: v for k, v in adapter_dict.items() if 'adapter' in k}
            vig_dict = self.vig_fusion.state_dict()
            vig_dict.update(adapter_dict)
            self.vig_fusion.load_state_dict(vig_dict)
        else:
            self.vig_fusion = ViGFusion(grasp_dim, args.width, args.layers, args.heads, self.device).to(device=self.device)

        self.feature_optim = Adam(self.vig_fusion.parameters(), lr=args.lr)

        # critic
        self.critic = QNetwork(args.width, args.hidden_size).to(device=self.device)            
        self.critic_target = QNetwork(args.width, args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # policy
        self.policy = Policy(args.width, args.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


        if not args.evaluate:
            # SAC parameters
            self.gamma = args.gamma
            self.tau = args.tau
            self.alpha = args.alpha # self.alpha = 0
            self.target_update_interval = args.target_update_interval
            self.automatic_entropy_tuning = args.automatic_entropy_tuning

            if self.automatic_entropy_tuning:
                # self.log_alpha = torch.tensor(0., requires_grad=True)
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
                self.alpha = self.log_alpha.exp()
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            # hard update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            
            # fix CLIP parameters and parameters of its adapter
            for k,v in self.vig_fusion.named_parameters():
                if 'clip' or 'adapter' in k:
                    v.requires_grad = False # fix parameters
                    # print(v.requires_grad)

            self.vig_fusion.train()
            self.critic.train()
            self.critic_target.train()
            self.policy.train()

        else:
            self.vig_fusion.eval()
            self.critic.eval()
            self.critic_target.eval()
            self.policy.eval()


    def get_fusion_feature(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions):
        vilg_feature, vig_attn, clip_match, clip_score = self.vig_fusion(bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions)
        return vilg_feature, vig_attn, clip_match, clip_score


    def select_action(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions, evaluate=False):
        sa, vig_attn, clip_match, clip_score = self.get_fusion_feature(bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions)
        logits = self.policy(sa)
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        mu = logits.argmax(-1)  # [B,]
        cate_dist = td.Categorical(logits=logits)
        pi = cate_dist.sample()  # [B,]

        action = pi if not evaluate else mu

        return logits.detach().cpu().numpy(), action.detach().cpu().numpy()[0], vig_attn.detach().cpu().numpy()[0], clip_match.detach().cpu().numpy()[0], clip_score.detach().cpu().numpy()[0]


    def select_action_w_obj(self, grasp_pose_set, pos_obj):
        pos_obj = np.array(pos_obj)[np.newaxis, :]
        pos_grasps = np.array(grasp_pose_set)[:, :3]
        dist = np.linalg.norm((pos_obj-pos_grasps), axis=1)
        action_idxs = np.where(dist<=0.07)[0]
        if action_idxs.shape[0] > 0:
            action_idx = np.random.choice(action_idxs) 
        else:
            action_idx = None
        
        return action_idx


    def forward(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions):
        sa, _, _, _ = self.get_fusion_feature(bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions)
        logits = self.policy(sa)
        
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)

        cate_dist = td.Categorical(logits=logits)
        pi = cate_dist.sample()  # [B,]
        log_prob = cate_dist.log_prob(pi).unsqueeze(-1)
        
        qf1, qf2 = self.critic(sa)
        return log_prob, qf1, qf2
        

    def update_parameters(self, memory, batch_size, steps_per_update, updates, qf_loss_list):

        # Sample a batch from memory
        target_bboxes_batch, target_pos_bboxes_batch, bboxes_batch, pos_bboxes_batch, grasps_batch, action_batch, reward_batch, mask_batch, next_bboxes_batch, next_pos_bboxes_batch, next_grasps_batch = memory.sample(batch_size=batch_size)

        target_bboxes_batch = torch.FloatTensor(target_bboxes_batch).to(self.device)
        bboxes_batch = torch.FloatTensor(bboxes_batch).to(self.device)
        grasps_batch = torch.FloatTensor(grasps_batch).to(self.device)
        target_pos_bboxes_batch = torch.FloatTensor(target_pos_bboxes_batch).to(self.device)
        pos_bboxes_batch = torch.FloatTensor(pos_bboxes_batch).to(self.device)
        next_bboxes_batch = torch.FloatTensor(next_bboxes_batch).to(self.device)
        next_pos_bboxes_batch = torch.FloatTensor(next_pos_bboxes_batch).to(self.device)
        next_grasps_batch = torch.FloatTensor(next_grasps_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            next_sa, _, _, _ = self.get_fusion_feature(next_bboxes_batch, next_pos_bboxes_batch, target_bboxes_batch, target_pos_bboxes_batch, next_grasps_batch)

            logits = self.policy(next_sa)
            if next_sa.shape[0] == 1:
                logits = logits.unsqueeze(0)
            if next_grasps_batch.shape[1] == 1:
                logits = logits.unsqueeze(0)
            # cate_dist = td.Categorical(logits=logits)
            # next_action = cate_dist.sample()  # [B,]
            # next_state_log_pi = cate_dist.log_prob(next_action).unsqueeze(-1)  # [B, 1]

            logits_prob = F.softmax(logits, -1)
            z = logits_prob == 0.0
            z = z.float() * 1e-8
            next_log_probs = torch.log(logits_prob + z)
            # next_log_probs = logits.log_softmax(-1)

            qf1_next_target, qf2_next_target = self.critic_target(next_sa) # [B, A, 1]
            qf1_next_target = qf1_next_target.reshape(qf1_next_target.shape[0], -1) # [B, A]
            qf2_next_target = qf2_next_target.reshape(qf2_next_target.shape[0], -1) # [B, A]

            v1_target = (next_log_probs.exp() * (qf1_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            v2_target = (next_log_probs.exp() * (qf2_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            min_qf_next_target = torch.min(v1_target, v2_target)

            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # next_q_value = torch.max(next_q_value.squeeze(-1), dim=1)[0]
        sa, _, _, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, target_bboxes_batch, target_pos_bboxes_batch, grasps_batch)
        qf1, qf2 = self.critic(sa)  # Two Q-functions to mitigate positive bias in the policy improvement step        
        # qf1 = torch.max(qf1.squeeze(-1), dim=1)[0]
        # qf2 = torch.max(qf2.squeeze(-1), dim=1)[0]

        qf1 = qf1.squeeze(-1)
        qf2 = qf2.squeeze(-1)
        qf1 = qf1.gather(1, action_batch.to(torch.int64))
        qf2 = qf2.gather(1, action_batch.to(torch.int64))

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = 0.5 * qf1_loss + 0.5 * qf2_loss
        qf_loss_list.append(qf_loss)
        
        # update critic with mean loss before n updates
        if (updates + 1) % steps_per_update == 0 and updates < 8000:
            # average loss
            qf_loss = sum(qf_loss_list) / len(qf_loss_list)
            self.critic_optim.zero_grad()
            qf_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
            self.critic_optim.step()
            qf_loss_list = []
        elif updates >= 8000:
            self.critic_optim.zero_grad()
            qf_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
            self.critic_optim.step()  
            qf_loss_list = []          

        if updates >= 8000:
            sa, _, _, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, target_bboxes_batch, target_pos_bboxes_batch, grasps_batch)
            logits = self.policy(sa)
            if sa.shape[0] == 1:
                logits = logits.unsqueeze(0)
            if grasps_batch.shape[1] == 1:
                logits = logits.unsqueeze(0)
            # cate_dist = td.Categorical(logits=logits)
            # action = cate_dist.sample()  # [B,]
            # log_pi = cate_dist.log_prob(action).unsqueeze(-1)  # [B, 1]
            logits_prob = F.softmax(logits, -1)
            z = logits_prob == 0.0
            z = z.float() * 1e-8
            log_probs = torch.log(logits_prob + z)
            # log_probs = logits.log_softmax(-1)
            entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

            qf1_pi, qf2_pi = self.critic(sa)
            # qf1_pi = torch.max(qf1_pi.squeeze(-1), dim=1)[0]
            # qf2_pi = torch.max(qf2_pi.squeeze(-1), dim=1)[0]
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # input should be a distribution in the log space
            # or cos similarity?
            # cossim_loss = logits @ clip_guided_logits.t()

            if self.automatic_entropy_tuning:
                self.alpha = self.log_alpha.exp()
            
            policy_loss = -((min_qf_pi - self.alpha * log_probs) * log_probs.exp()).sum(-1).mean()
            # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                self.alpha = self.log_alpha.exp()
                self.target_entropy = 0.98 * -np.log(1 / grasps_batch.shape[1])
                # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                corr = (self.target_entropy - entropy).detach()
                alpha_loss = -(self.alpha * corr).mean()

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

        # !!! raw version update feature module with total loss !!! 
        # TODO maybe update critic, policy and feature block together, but might be unstable?
        if updates >= 8000:
            with torch.no_grad():
                next_sa, _, _, _ = self.get_fusion_feature(next_bboxes_batch, next_pos_bboxes_batch, target_bboxes_batch, target_pos_bboxes_batch, next_grasps_batch)
                
                logits = self.policy(next_sa)
                if next_sa.shape[0] == 1:
                    logits = logits.unsqueeze(0)
                if next_grasps_batch.shape[1] == 1:
                    logits = logits.unsqueeze(0)
                # cate_dist = td.Categorical(logits=logits)
                # next_action = cate_dist.sample()  # [B,]
                # next_state_log_pi = cate_dist.log_prob(next_action).unsqueeze(-1)  # [B, 1]
                next_log_probs = logits.log_softmax(-1)

                qf1_next_target, qf2_next_target = self.critic_target(next_sa) # [B, A, 1]

                qf1_next_target = qf1_next_target.reshape(qf1_next_target.shape[0], -1) # [B, A]
                qf2_next_target = qf2_next_target.reshape(qf2_next_target.shape[0], -1) # [B, A]

                v1_target = (next_log_probs.exp() * (qf1_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
                v2_target = (next_log_probs.exp() * (qf2_next_target - self.alpha * next_log_probs)).sum(-1, keepdim=True)
                min_qf_next_target = torch.min(v1_target, v2_target)

                # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
                next_q_value = next_q_value[0]
                # next_q_value = torch.max(next_q_value.squeeze(-1), dim=1)[0]
                
            sa, _, _, _ = self.get_fusion_feature(bboxes_batch, pos_bboxes_batch, target_bboxes_batch, target_pos_bboxes_batch, grasps_batch)
            qf1, qf2 = self.critic(sa)  # Two Q-functions to mitigate positive bias in the policy improvement step        
            qf1 = torch.max(qf1.squeeze(-1), dim=1)[0]
            qf2 = torch.max(qf2.squeeze(-1), dim=1)[0]
            qf1_loss_ = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss_ = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            logits = self.policy(sa)
            if sa.shape[0] == 1:
                logits = logits.unsqueeze(0)
            if grasps_batch.shape[1] == 1:
                logits = logits.unsqueeze(0)
            # cate_dist = td.Categorical(logits=logits)
            # action = cate_dist.sample()  # [B,]
            # log_pi = cate_dist.log_prob(action).unsqueeze(-1)  # [B, 1]
            log_probs = logits.log_softmax(-1)

            qf1_pi, qf2_pi = self.critic(sa)
            # qf1_pi = torch.max(qf1_pi.squeeze(-1), dim=1)[0]
            # qf2_pi = torch.max(qf2_pi.squeeze(-1), dim=1)[0]
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            if self.automatic_entropy_tuning:
                self.alpha = self.log_alpha.exp()
            policy_loss_ = -((min_qf_pi - self.alpha * log_probs) * log_probs.exp()).sum(-1).mean()
            # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            total_loss = 0.5 * (0.5 * qf1_loss_ + 0.5 * qf2_loss_) + 0.5 * policy_loss_

            self.feature_optim.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.vig_fusion.parameters(), 0.1)
            self.feature_optim.step()
        
        if updates >= 8000:
            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), total_loss.item(), qf_loss_list
        else:
            return qf1_loss.item(), qf2_loss.item(), 0., 0., 0., 0., qf_loss_list

# network definition
class ViGR(nn.Module):
    def __init__(self, grasp_dim, args):
        super().__init__()
        self.device = args.device
        if args.use_adapter:
            self.vig_fusion = ViGFusion_Adapter(grasp_dim, args.width, args.layers, args.heads, self.device).to(device=self.device)
            # load adapter parameters
            adapter_dict = torch.load(args.clip_model_path)
            adapter_dict = {k: v for k, v in adapter_dict.items() if 'adapter' in k}
            vig_dict = self.vig_fusion.state_dict()
            vig_dict.update(adapter_dict)
            self.vig_fusion.load_state_dict(vig_dict)
        else:
            self.vig_fusion = ViGFusion(grasp_dim, args.width, args.layers, args.heads, self.device).to(device=self.device)
        
        self.policy = Policy(args.width, args.hidden_size).to(self.device)
        # self.obj_policy = Policy(args.width, args.hidden_size).to(self.device)

        for k,v in self.vig_fusion.named_parameters():
            if 'clip' or 'adapter' in k:
                v.requires_grad = False # fix parameters
                # print(v.requires_grad)

    def forward(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions):
        sa, _, _, _ = self.vig_fusion(bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions)
        logits = self.policy(sa)
        if sa.shape[0] == 1:
            logits = logits.unsqueeze(0)
        if actions.shape[1] == 1:
            logits = logits.unsqueeze(0)
        action = logits.argmax(-1)  # [B,]

        return logits, action

    def save_rearrange_checkpoint(self, model, ckpt_path):
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'feature_state_dict': model.vig_fusion.state_dict(),
                    'policy_state_dict': model.policy.state_dict()}, ckpt_path)

