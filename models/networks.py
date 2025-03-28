from cmath import sin
from collections import OrderedDict
from dataclasses import dataclass
import imp
import math
import numpy as np
from copy import deepcopy
from tkinter import E
import kornia as K
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Normal
from models.core.clip import build_model, load_clip, tokenize, AttentionPool2d
from models.core.resnet import resnet50

############################################# Basic Blocks ################################################
# positional embedding with sin/cos
class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights

# multi layer
class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        for i, _ in enumerate(self.resblocks):
            q, attn_weights = self.resblocks[i](q, k, v)

        q = q.permute(1, 0, 2) # L'ND -> NL'D
        return q, attn_weights

# one layer without shortcut: naivest cross attention
class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=1024, num_heads=32, output_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, q, k, v, attn_mask=None):
        x, attn_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask
        )
        
        return x, attn_weights


############################################# Visual-grasping Fusion Block ################################################
class ViGFusion(nn.Module):
    def __init__(self, grasp_dim, width, layers, heads, device):
        super().__init__()
        
        self.device = device
        self._load_clip()
        
        self.visual_cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)
        self.visual_grasp_cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)

        self.grasp_embbedding = nn.Sequential(
                                nn.Linear(grasp_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)

        self.bbox_pos_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
                            
    def _load_clip(self): # patch_size = 32, vision_width = 768, vision_layers = 12, vision_heads = 12, output_dim = 512, vocab_size = 49408
        model, _ = load_clip("ViT-B/32", device=self.device)
        self.clip = build_model(model.state_dict()).to(self.device)
        del model

    def encode_bbox(self, x):
        self.clip.eval()
        with torch.no_grad():
            if x.shape[0] == 1:
                bboxs = x[0]
                padding = nn.ZeroPad2d(bboxs[0].shape[1] * 3)
                bboxs = padding(bboxs)
                bbox_feat = self.clip.encode_image(bboxs.to(self.device))
                bbox_feat = bbox_feat.unsqueeze(0)
        return bbox_feat

    def encode_grasp(self, x):
        grasp_emb = self.grasp_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_bbox_pos(self, x):
        bbox_pos_emb = self.bbox_pos_embbedding(x.to(self.device)) # shape = [N, L', D]
        return bbox_pos_emb

    def forward(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions):
        # encode current bboxes and positions
        bbox_feat = self.encode_bbox(bboxes) # shape = [N, L1, D] D=512
        pos_bboxes = self.pos_projection(pos_bboxes)
        bbox_pos_feat = self.encode_bbox_pos(pos_bboxes) # shape = [N, L1, D]
        
        # add fusion
        bbox_compound_feat = bbox_pos_feat + bbox_feat
        # concat fusion
        # bbox_compound_feat = torch.cat((bbox_feat, bbox_pos_feat), dim=-1)
        # bbox_compound_feat = self.pos_vision_fusion(bbox_compound_feat)
        bbox_compound_feat = bbox_compound_feat.permute(1, 0, 2) # NLD -> LND
        
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L2, D] D=512
        target_pos_bboxes = self.pos_projection(target_pos_bboxes)
        target_bbox_pos_feat = self.encode_bbox_pos(target_pos_bboxes) # shape = [N, L2, D]      
        # add fusion
        target_bbox_compound_feat = target_bbox_pos_feat + target_bbox_feat
        target_bbox_compound_feat = target_bbox_compound_feat.permute(1, 0, 2) # NLD -> LND

        # get clip match scores
        # normalized features   
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t() # shape = [N, L1, L2]

        # softmax by each bbox
        probs = logits_per_image.softmax(dim=-1)
        
        clip_match = torch.max(probs, dim=-1)[1]
        clip_score = torch.max(probs, dim=-1)[0]

        # encode grasp
        grasp_feat = self.encode_grasp(actions) # shape = [N, L', D]
        grasp_feat = grasp_feat.permute(1, 0, 2)  # NL'D -> L'ND

        # cross attention
        # cross_feat, attn_weights = self.cross_attn(q=grasp_feat, k=bbox_pos_feat, v=fusion_feat) # shape = [N, L', D]
        # Note: attn weights, 
        visual_fusion_feat, _ = self.visual_cross_attn(q=bbox_compound_feat, k=target_bbox_compound_feat, v=target_bbox_compound_feat) # shape = [N, L1, D]
        visual_fusion_feat = visual_fusion_feat.permute(1, 0, 2) # NLD -> LND

        cross_feat, attn_weights = self.visual_grasp_cross_attn(q=grasp_feat, k=visual_fusion_feat, v=visual_fusion_feat) # shape = [N, L', D]
        
        return cross_feat, attn_weights, clip_match, clip_score

class ViGFusion_Adapter(nn.Module):
    def __init__(self, grasp_dim, width, layers, heads, device):
        super().__init__()
        
        self.device = device
        self._load_clip()
        self.adapter = Adapter(512, 4).to(self.clip.dtype).to(self.device)
        
        self.visual_cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)
        self.visual_grasp_cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)
        # self.visual_self_attn = CrossTransformer(width=width, layers=layers, heads=heads)

        self.grasp_embbedding = nn.Sequential(
                                nn.Linear(grasp_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)

        self.bbox_pos_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
                            
    def _load_clip(self): # patch_size = 32, vision_width = 768, vision_layers = 12, vision_heads = 12, output_dim = 512, vocab_size = 49408
        model, _ = load_clip("ViT-B/32", device=self.device)
        self.clip = build_model(model.state_dict()).to(self.device)
        del model


    def encode_bbox(self, x):
        self.clip.eval()
        with torch.no_grad():
            if x.shape[0] == 1:
                bboxs = x[0]
                padding = nn.ZeroPad2d(bboxs[0].shape[1] * 3)
                bboxs = padding(bboxs)
                bbox_feat = self.clip.encode_image(bboxs.to(self.device))
                bbox_feat = bbox_feat.unsqueeze(0)
        #     bbox_feat_concated = None
        #     for batch in range(x.shape[0]):
        #         bboxes = x[batch]
        #         padding = nn.ZeroPad2d(bboxes.shape[-1] * 3)
        #         bboxes = padding(bboxes)
        #         bbox_feat = self.clip.encode_image(bboxes.to(self.device))
        #         bbox_feat = bbox_feat.unsqueeze(0)
        #         if bbox_feat_concated is None:
        #             bbox_feat_concated = bbox_feat
        #         else:
        #             bbox_feat_concated = torch.cat((bbox_feat_concated, bbox_feat), dim=0)
        # return bbox_feat_concated
        return bbox_feat

    def encode_grasp(self, x):
        grasp_emb = self.grasp_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_bbox_pos(self, x):
        bbox_pos_emb = self.bbox_pos_embbedding(x.to(self.device)) # shape = [N, L', D]
        return bbox_pos_emb

    def forward(self, bboxes, pos_bboxes, target_bboxes, target_pos_bboxes, actions):
        # minimize the distance between clip probability distribution and grasp affordance distribution
        # if actions.shape[0] == 1:
        #     pos_bboxes = pos_bboxes[0].unsqueeze(1)
        #     pose_grasps = actions[0].unsqueeze(0)
        #     pos_grasps = pose_grasps[:, :, :3]
        #     dist_map = torch.norm((pos_bboxes-pos_grasps), dim=2)
        #     bbox_grasp_map = (dist_map<0.05).float()

        # adapter ratio
        ratio = 0.2

        # encode current bboxes and positions
        bbox_feat = self.encode_bbox(bboxes) # shape = [N, L1, D] D=512
        bbox_feat_adapter = self.adapter(bbox_feat)
        bbox_feat = ratio * bbox_feat_adapter + (1 - ratio) * bbox_feat
        pos_bboxes = self.pos_projection(pos_bboxes)
        bbox_pos_feat = self.encode_bbox_pos(pos_bboxes) # shape = [N, L1, D]
        # add fusion
        bbox_compound_feat = bbox_pos_feat + bbox_feat
        bbox_compound_feat = bbox_compound_feat.permute(1, 0, 2) # NLD -> LND
        
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L2, D] D=512
        target_bbox_feat_adapter = self.adapter(target_bbox_feat)        
        target_bbox_feat = ratio * target_bbox_feat_adapter + (1 - ratio) * target_bbox_feat
        target_pos_bboxes = self.pos_projection(target_pos_bboxes)
        target_bbox_pos_feat = self.encode_bbox_pos(target_pos_bboxes) # shape = [N, L2, D]      
        # add fusion
        target_bbox_compound_feat = target_bbox_pos_feat + target_bbox_feat
        target_bbox_compound_feat = target_bbox_compound_feat.permute(1, 0, 2) # NLD -> LND

        # get clip match scores
        # normalized features   
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t() # shape = [N, L1, L2]

        # softmax by each bbox
        probs = logits_per_image.softmax(dim=-1)
        
        clip_match = torch.max(probs, dim=-1)[1]
        clip_score = torch.max(probs, dim=-1)[0]

        # encode grasp
        grasp_feat = self.encode_grasp(actions) # shape = [N, L', D]
        grasp_feat = grasp_feat.permute(1, 0, 2)  # NL'D -> L'ND

        # cross attention of current and goal objects
        # cross_feat, attn_weights = self.cross_attn(q=grasp_feat, k=bbox_pos_feat, v=fusion_feat) # shape = [N, L', D]
        visual_fusion_feat, _ = self.visual_cross_attn(q=bbox_compound_feat, k=target_bbox_compound_feat, v=target_bbox_compound_feat) # shape = [N, L1, D]
        visual_fusion_feat = visual_fusion_feat.permute(1, 0, 2) # NLD -> LND

        # self-attention of current and goal object attentions
        # visual_fusion_feat, _ = self.visual_self_attn(q=visual_fusion_feat, k=visual_fusion_feat, v=visual_fusion_feat) # shape = [N, L1, D]
        # visual_fusion_feat = visual_fusion_feat.permute(1, 0, 2) # NLD -> LND
        
        cross_feat, attn_weights = self.visual_grasp_cross_attn(q=grasp_feat, k=visual_fusion_feat, v=visual_fusion_feat) # shape = [N, L', D]
        
        return cross_feat, attn_weights, clip_match, clip_score

############################################# Active Seeing Block ################################################
class ASBlock(nn.Module):
    def __init__(self, args):
        super().__init__()  

        self.device = args.device
        self.matcher = args.matcher

        # matching module
        if self.matcher == "CLIP":
            # CLIP
            self._load_clip()
        
        #resnet feature module
        self.resnet = resnet50(input_dim=2)
        modules = list(self.resnet.children())
        self.stem = modules[0]
        self.layer1 = modules[1]
        self.layer2 = modules[2]
        self.layer3 = modules[3]
        self.layer4 = modules[4]
        self.pooling = modules[5]

    def _load_clip(self): # patch_size = 32, vision_width = 768, vision_layers = 12, vision_heads = 12, output_dim = 512, vocab_size = 49408
        model, _ = load_clip("ViT-B/32", device=self.device)
        self.clip = build_model(model.state_dict()).to(self.device)
        del model


    def encode_bbox(self, x):
        self.clip.eval()
        with torch.no_grad():
            if x.shape[0] == 1:
                bboxs = x[0]
                padding = nn.ZeroPad2d(bboxs[0].shape[1] * 3)
                bboxs = padding(bboxs)
                bbox_feat = self.clip.encode_image(bboxs.to(self.device))
                bbox_feat = bbox_feat.unsqueeze(0)
        return bbox_feat
            
    def get_clip_match_dist(self, bbox, target_bboxes): # bbox shape = [N, 1, 3, H, W], target_bboxes shape = [N, L, 3, H, W]
        # ------------------------- CLIP ------------------------ #
        # encode current bbox
        bbox_feat = self.encode_bbox(bbox) # shape = [N, 1, D] D=512
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L, D] D=512
        # get clip match scores
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t()

        # softmax by each bbox
        prob = logits_per_image.softmax(dim=-1)
        match = torch.max(prob, dim=-1)[1]
        score = torch.max(prob, dim=-1)[0]
        z = prob == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(prob + z)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

        return prob, match, score, entropy

    def resnet50_feature(self, x):
        im = []
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4, self.pooling]:
            x = layer(x)
            im.append(x)
        return x, im

    def forward(self, delta_flow):

        # delta flow to represent matching uncertainty
        flow_feat, _ = self.resnet50_feature(delta_flow.to(self.device)) # [1, 2048, 1, 1]
        flow_feat = flow_feat.view(flow_feat.size(0), -1) # [1, 2048]

        return flow_feat

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CustomCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self._load_clip()

        self.adapter = Adapter(512, 4).to(self.clip.dtype).to(self.device)

    def _load_clip(self): # patch_size = 32, vision_width = 768, vision_layers = 12, vision_heads = 12, output_dim = 512, vocab_size = 49408
        model, _ = load_clip("ViT-B/32", device=self.device)
        self.clip = build_model(model.state_dict()).to(self.device)
        del model         
   
    def encode_bbox(self, x):
        self.clip.eval()
        with torch.no_grad():
            if x.shape[0] == 1:
                bboxs = x[0]
                padding = nn.ZeroPad2d(bboxs[0].shape[1] * 3)
                bboxs = padding(bboxs)
                bbox_feat = self.clip.encode_image(bboxs.to(self.device))
                bbox_feat = bbox_feat.unsqueeze(0)
        return bbox_feat

    def get_clip_match_dist(self, bbox, target_bboxes): # bbox shape = [N, 1, 3, H, W], target_bboxes shape = [N, L, 3, H, W]
        # ------------------------- CLIP ------------------------ #
        # encode current bbox
        bbox_feat = self.encode_bbox(bbox) # shape = [N, 1, D] D=512
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L, D] D=512
        # get clip match scores
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t()

        # softmax by each bbox
        prob = logits_per_image.softmax(dim=-1)
        match = torch.max(prob, dim=-1)[1]
        score = torch.max(prob, dim=-1)[0]
        z = prob == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(prob + z)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

        return prob, match, score, entropy

    def get_customclip_match_dist(self, bbox, target_bboxes): # bbox shape = [N, 1, 3, H, W], target_bboxes shape = [N, L, 3, H, W]
        # ------------------------- CLIP ------------------------ #
        # encode current bbox
        bbox_feat = self.encode_bbox(bbox) # shape = [N, 1, D] D=512
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L, D] D=512
   
        bbox_feat_adapter = self.adapter(bbox_feat)
        target_bbox_feat_adapter = self.adapter(target_bbox_feat)

        ratio = 0.2
        bbox_feat = ratio * bbox_feat_adapter + (1 - ratio) * bbox_feat
        target_bbox_feat = ratio * target_bbox_feat_adapter + (1 - ratio) * target_bbox_feat

        # get clip match scores
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t()

        # softmax by each bbox
        prob = logits_per_image.softmax(dim=-1)
        match = torch.max(prob, dim=-1)[1]
        score = torch.max(prob, dim=-1)[0]
        z = prob == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(prob + z)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True)

        return prob, match, score, entropy
                
    def forward(self, bbox, target_bboxes): # bbox shape = [N, 1, 3, H, W], target_bboxes shape = [N, L, 3, H, W]
        # ------------------------- CLIP ------------------------ #
        # encode current bbox
        bbox_feat = self.encode_bbox(bbox) # shape = [N, 1, D] D=512
        # encode target bboxes and positions
        target_bbox_feat = self.encode_bbox(target_bboxes) # shape = [N, L, D] D=512
   
        bbox_feat_adapter = self.adapter(bbox_feat)
        target_bbox_feat_adapter = self.adapter(target_bbox_feat)

        ratio = 0.2
        bbox_feat = ratio * bbox_feat_adapter + (1 - ratio) * bbox_feat
        target_bbox_feat = ratio * target_bbox_feat_adapter + (1 - ratio) * target_bbox_feat

        # get clip match scores
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        target_bbox_feat_normlized = target_bbox_feat / target_bbox_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ target_bbox_feat_normlized.squeeze().t()

        return logits_per_image
 
############################################# Actor-critic Block ################################################
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Q Networks for RL algorithms (discrete action)
class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, sa):
        
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Policy Network for RL algorithms (discrete action)
class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x).squeeze()
        return logits

# Q Networks for RL algorithms (continuous action)
class QNetwork_(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_dim):
        super(QNetwork_, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Policy Network for RL algorithms (continuous action)
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        self.log_sig_max = 2
        self.log_sig_min = -20
        self.epsilon = 1e-6

        # For pose, action space is [-3.14, 3.14]
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space[1] - action_space[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space[1] + action_space[0]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

# Policy Network for RL algorithms (continuous action)
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.noise = torch.Tensor(action_dim)

        self.apply(weights_init_)

        # For pose, action space is [-3.14, 3.14]
        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space[1] - action_space[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space[1] + action_space[0]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

# Policy Network of Gaussian Mixture Model
class GMMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_modes=5, min_std=0.0001):
        super(GMMPolicy, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, output_dim * num_modes)
        self.logstd_layer = nn.Linear(hidden_dim, output_dim * num_modes)
        self.logits_layer = nn.Linear(hidden_dim, num_modes)
        
        self.apply(weights_init_)

        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std


    def forward_fn(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        means = torch.tanh(means)
        logits = self.logits_layer(x)

        logstds = self.logstd_layer(x).view(-1, self.num_modes, self.output_dim)
        stds = torch.exp(logstds) + self.min_std
        return means, stds, logits

    def forward(self, x):
        means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        
        return gmm