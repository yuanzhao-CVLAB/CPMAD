import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from mymodels.basetrainer import BaseTrainer
from mymodels.CPL_Modules import Mlp, DropPath

class CrossModalAnchorGenerator(nn.Module):
    

    def __init__(self, d_model):
        super().__init__()
        self.proj_rgb = Mlp(d_model, d_model * 2, d_model)
        self.proj_depth = Mlp(d_model, d_model * 2, d_model)
        self.proj_concat = Mlp(d_model * 2, d_model, d_model)
        self.proj_anchor = Mlp(d_model, d_model * 2, d_model * 2)
        self.norm = nn.LayerNorm(d_model)
        self.out = Mlp(d_model, d_model * 2, d_model)

    def forward(self, anchors, rgb, depth):
        rgb = rgb + self.proj_rgb(rgb)
        depth = depth + self.proj_depth(depth)
        rgb_anchors, depth_anchors = self.proj_anchor(anchors).chunk(2, dim=-1)
        V = self.proj_concat(torch.concat([rgb, depth], -1))

        rgb = F.normalize(rgb, dim=-1)
        depth = F.normalize(depth, dim=-1)
        rgb_anchors = F.normalize(rgb_anchors, dim=-1)
        depth_anchors = F.normalize(depth_anchors, dim=-1)

        sim_rgb = torch.matmul(rgb_anchors, rgb.transpose(1, 2))
        sim_depth = torch.matmul(depth_anchors, depth.transpose(1, 2))
        sim = (sim_rgb + sim_depth) / 2
        weights = sim

        anchors = anchors + torch.matmul(weights, V) / (weights.sum(-1).unsqueeze(-1) + 1e-6)
        anchors = anchors + self.norm(self.out(anchors))
        return anchors


class CEM(nn.Module):
    """
    Cross-Modal Consensus Module
    """

    def __init__(self, D: int, K: int = 12,
                ):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(1, K, D))
        self.anchor_gen = CrossModalAnchorGenerator(D)
        self.cross_attn = nn.MultiheadAttention(D, 8, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.out = Mlp(D, D * 2, D)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        rgb = torch.cat(rgb, 1)
        depth = torch.cat(depth, 1)

        x = rgb + depth
        anchors = self.anchor.repeat((rgb.shape[0], 1, 1))

        prototype = self.anchor_gen(anchors, rgb, depth)

        consensus, _ = self.cross_attn(x, prototype, prototype)

        consensus = consensus + self.norm(self.out(consensus))
        consensus = consensus.unflatten(1, (2, -1)).mean(1)
        return consensus  



class Dino(nn.Module):
    def __init__(self, backbone="dinov2reg_vit_base_14"):
        super().__init__()
        if "base" in backbone or "small" in backbone:
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
            self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        elif "large" in backbone:
            self.target_layers = list(range(3, 19))
            self.fuse_layer_encoder = [list(range(0, 8)), list(range(8, 16))]

        self.act = nn.ReLU()
        self.num_register_tokens = 4

        self.encoder = timm.create_model(model_name=backbone,
                                         pretrained_cfg_overlay=dict(
                                             file=f"checkpoints/{backbone}.safetensors"),
                                         pretrained=True,
                                         **{'features_only': True,
                                            'out_indices': self.target_layers})
        self.embed_dim = self.encoder.model.embed_dim

    def forward(self, x):
        return self.forward_timm(x)

    def forward_timm(self, x):
        f = self.encoder(x)
        en = [self.fuse_feature([f[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        out = [self.act(o) for o in en]
        out = [o.flatten(-2, -1).permute(0, 2, 1) for o in out]
        return out

    def forward_dino(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:

                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        global_list = [e[:, :1 + self.encoder.num_register_tokens, :] for e in en_list]
        global_list = [self.fuse_feature([global_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        out = [self.act(o) for o in en]
        return out, en_list

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class CRAttention(nn.Module):
 

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可学习的缩放因子来平衡两路 logits 的贡献
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.ln = nn.LayerNorm(dim)

    def _shape_qkv(self, x, linear):
        B, L, C = x.shape
        y = linear(x)
        y = y.view(B, L, self.num_heads, self.head_dim)
        y = y.permute(0, 2, 1, 3).contiguous()
        return y

    def forward(self, specific, feature, share):
        """
        Args:
          specific: (B, T, C)  -- query tokens (e.g., region tokens)
          feature:  (B, N, C)  -- feature tokens (one key source + value source)
          share:    (B, M, C)  -- shared tokens (other key source)
        Returns:
          out: (B, T, C)
        """
        B, T, C = specific.shape
        _, N, _ = feature.shape
        _, M, _ = share.shape
        q1 = self._shape_qkv(specific, self.q1)
        k1 = self._shape_qkv(share, self.k1)

        q2 = self._shape_qkv(specific, self.q2)
        k2 = self._shape_qkv(feature, self.k2)

        logits1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        logits2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        assert logits1.shape[-1] == logits2.shape[-1], "shape keep consistent"

        combined_logits = self.alpha * logits1 - self.beta * logits2
        attn = F.softmax(combined_logits, dim=-1)

        attn = self.attn_drop(attn)

        v = self._shape_qkv(feature, self.v)
        K = attn.shape[-1]
        if K != v.shape[-2]:
            raise RuntimeError(
                f"Attention key length {K} != value length {v.shape[-2]}; align share/feature key counts.")

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        out = self.proj_drop(self.proj(out))
        out = self.ln(out + specific)
        return out, attn


class SQM(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CRAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, specific, feature, share, return_attention=True):
        x = specific
        y, attn = self.attn(self.norm1(x), self.norm1(feature), self.norm1(share))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class CMappingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        x, attn = self.attn(x, prototype_token, prototype_token)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CMapping(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CMappingAttention(
            dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.global_semantic = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, prototype, return_attention=False):
        shortcut = self.global_semantic(x).mean(1, keepdim=True)
        y, attn = self.attn(x, prototype)
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x))) + shortcut
        if return_attention:
            return x, attn
        else:
            return x


class CPLBlock(nn.Module):
   

    def __init__(self, feature_dim, num_queries=8, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim

        # 常态概念查询向量
        self.normality_queries = nn.Parameter(torch.randn(num_queries, feature_dim))

        self.prototype_extractor = SQM(feature_dim, 8)
        self.prototype_block = CMapping(feature_dim, 8)

    def forward(self, feature, share):
        B = share.shape[0]

        protype, attn_SPQ = self.prototype_extractor(self.normality_queries.unsqueeze(0).repeat(B, 1, 1), feature,
                                                     share)
        out, attn_CM = self.prototype_block(share, protype, return_attention=True)

        return out, attn_SPQ, attn_CM


class CPL(nn.Module):
    def __init__(self, embed_dim, num_private_prototypes):
        super().__init__()

        self.extractor_rgb = CPLBlock(embed_dim, num_private_prototypes)
        self.extractor_depth = CPLBlock(embed_dim, num_private_prototypes)

    def forward(self, fuse_shared, f_rgb, f_depth):
        p_shared = fuse_shared

        f_recon_rgb, attn_SPQ_rgb, attn_CM_rgb = self.extractor_rgb(f_rgb, p_shared)
        f_recon_depth, attn_SPQ_depth, attn_CM_depth = self.extractor_depth(f_depth, p_shared)
        return f_rgb, f_depth, f_recon_rgb, f_recon_depth, p_shared

 

class CPLTrainer(BaseTrainer):
    def __init__(self, args, lambda_mi=0.1, top_k_ratio=0.01):
        super().__init__()

        self.backbone_rgb =  Dino(args["backbone"]).cuda()
        self.backbone_depth = Dino(args["backbone"]).cuda()
        dim = self.backbone_rgb.embed_dim
        self.model = nn.ModuleList(
            [CPL(dim, args["consensus_prototype_num"]) for _ in range(len(self.backbone_rgb.fuse_layer_encoder))])

        self.normality_bottleneck = CEM(dim, args["common_codebook_size"])

        # 指定需要训练的模块
        # PEFT会自动处理骨干网络的参数冻结
        self.trainable_layer = (
            "self.model.backbone_rgb, self.model.backbone_depth, "
            "self.model.fusion_module, self.model.query_module"
        )

    def cosine_sim(self, f_recon_rgb, f_rgb, f_recon_depth, f_depth):
        rgb_amap = 1 - F.cosine_similarity(f_recon_rgb, f_rgb, dim=-1)
        rgb_loss = rgb_amap.mean()
        depth_amap = 1 - F.cosine_similarity(f_recon_depth, f_depth, dim=-1)
        depth_loss = depth_amap.mean()
        return rgb_amap + depth_amap, rgb_loss + depth_loss

    def forward_step(self, batch, train=False):
        device = next(self.model.parameters()).device
        rgb_image = batch["RGB"].to(device)
        depth_image = batch["Depth"].to(device)
        with torch.no_grad():  # 主干网络大部分参数被冻结
            # DINOv2 输出 patch tokens 和 token, 我们只取 patch tokens
            local_rgb = self.backbone_rgb(rgb_image)
            local_depth = self.backbone_depth(depth_image)
        # 模型前向传播

        # 1. 计算一致性损失
        loss_cos = 0
        anomaly_map = 0

        share_feature = self.normality_bottleneck(local_rgb, local_depth)
        for i in range(len(local_rgb)):
            f_rgb, f_depth, f_recon_rgb, f_recon_depth, p_shared = self.model[i](share_feature, local_rgb[i],
                                                                                 local_depth[i])

            sim_map, sim_loss = self.cosine_sim(f_recon_rgb, f_rgb, f_recon_depth, f_depth)
            anomaly_map += sim_map
            loss_cos += sim_loss
        if train:
            # 3. 计算总损失
            loss_dict = {
                "Total_Loss": loss_cos.item(),
            }
            return loss_cos, loss_dict
        else:  # 推理/评估
            H = W = int(math.sqrt(anomaly_map.shape[-1]))
            anomaly_map = anomaly_map.view(1, 1, H, W)
            # 2. 上采样到原始图像尺寸
            anomaly_map_resized = F.interpolate(
                anomaly_map,
                size=rgb_image.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            return anomaly_map_resized

    def train_step(self, batch, **kwargs):
        self.model.train()

        total_loss, loss_details = self.forward_step(batch, train=True)

        return total_loss, loss_details

    def forward(self, batch):
        return self.eval_step(batch)

    def eval_step(self, batch, **kwargs):
        self.model.eval()
        anomaly_map_resized = self.forward_step(batch, train=False)

        return anomaly_map_resized, anomaly_map_resized

    def get_models(self):
        # 返回需要训练的模型，以便优化器进行参数更新
        return ("model,normality_bottleneck".split(","), self.model, self.normality_bottleneck)