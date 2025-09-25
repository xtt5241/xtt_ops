# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.ops import point_sample
from mmengine.dist import all_reduce
from mmengine.model.weight_init import (caffe2_xavier_init, normal_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, MatchMasks, SampleList,
                         seg_data_to_instance_data)
from mmseg.models.utils import (MLP, LayerNorm2d, PatchEmbed, cross_attn_layer,
                     get_uncertain_point_coords_with_randomness, resize)
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from dao import DAO
from einops import rearrange


class CrossAttention(nn.Module):
# Q（查询）：来自模型的 SideAdapterNetwork，即DAN 模块 生成的视觉 token（针对全景图畸变优化后的特征）；
# K（键）、V（值）：来自 “冻结 CLIP 模型的中间层视觉特征”（针孔图预训练的通用视觉知识）；
# 这种设计专门用于解决论文中的跨域特征适配问题
# 将 CLIP 在针孔图上学习的通用视觉知识，迁移到侧网络针对全景图的畸变适配特征中，为后续掩码生成和类别预测提供高质量融合特征。

    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.  # 输入特征维度（如CLIP特征维度768）
        num_heads (int): Number of attention heads. Default: 8  # 注意力头数
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v. Default: False.  # q/k/v是否加偏置
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.  # qk缩放因子
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0  # 注意力权重dropout
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0  # 输出dropout
        attn_head_dim (int, optional): Dimension of attention head.  # 单头注意力维度（默认dim//num_heads）
        out_dim (int, optional): Dimension of output.  # 输出维度（默认与输入dim一致）
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        #  CLIP 视觉特征的维度为 768（如 ViT-B/16），而侧网络（SideAdapterNetwork）的嵌入维度为 240
        #  通过out_dim参数支持输出维度自定义，直接将融合后的特征维度对齐到侧网络的 240 维，避免额外的维度转换层，减少计算开销
        if out_dim is None:
            out_dim = dim  # 若未指定输出维度，默认与输入一致（保证特征维度不突变）
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim  # 允许手动指定单头维度（灵活适配CLIP特征）
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 注意力缩放因子（默认1/√头维度，避免梯度消失）
        assert all_head_dim == dim, "输入dim必须等于头数×单头维度（保证注意力拆分有效）"

        # 定义q/k/v线性投影层（q对应侧网络特征，k/v对应CLIP特征）
        self.q = nn.Linear(out_dim, all_head_dim, bias=False)  #  query投影（侧网络特征→注意力维度）
        self.k = nn.Linear(dim, all_head_dim, bias=False)      #  key投影（CLIP特征→注意力维度）
        self.v = nn.Linear(dim, all_head_dim, bias=False)      #  value投影（CLIP特征→注意力维度）

        # 可选偏置（若启用，为q/k/v分别添加可学习偏置）
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        # Dropout层（正则化，避免过拟合）
        # attn_drop：对注意力权重矩阵进行 Dropout（默认 0.0，可根据过拟合情况调整），避免模型过度依赖 CLIP 特征中的特定 token；
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)  # 注意力输出投影回目标维度
        # proj_drop：对注意力输出的线性投影结果进行 Dropout（默认 0.0），进一步增强模型泛化能力，确保在全景图这种畸变场景下的鲁棒性。
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape  # x: [批次大小B, 侧网络特征序列长度N, 特征维度C]
        N_k = k.shape[1]   # k: CLIP特征序列长度（H×W，H/W为特征图高宽）
        N_v = v.shape[1]   # v: 与k同长度（CLIP特征序列）

        # 处理偏置（若启用，添加到q/k/v中）
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        # q/k/v线性投影（添加偏置）
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)  # [B, N, all_head_dim]
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)  # [B, N_k, all_head_dim]
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)  # [B, N_v, all_head_dim]

        # 拆分注意力头（适应多头注意力计算：[B, N, all_head_dim] → [B, num_heads, N, head_dim]）
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        # 计算注意力权重（q×k^T，缩放后softmax）
        q = q * self.scale  # 缩放（避免数值过大）
        attn = (q @ k.transpose(-2, -1))  # [B, num_heads, N, N_k]（注意力得分矩阵）
        attn = attn.softmax(dim=-1)       # 按k维度softmax（归一化注意力权重）
        attn = self.attn_drop(attn)       # 注意力权重dropout

        # 注意力加权求和（attn×v）+ 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # 合并注意力头：[B, N, all_head_dim]
        x = self.proj(x)                                   # 投影回目标维度：[B, N, out_dim]
        x = self.proj_drop(x)                              # 输出dropout
        return x


class MLPMaskDecoder(nn.Module):
# 核心功能是通过 MLP 层分别解码 “查询 tokens（query）” 和 “视觉特征（x）”这两种核心特征
# 生成论文中关键的两大输出：mask proposals，用于像素级分割定位 和attention bias，用于后续 CLIP 层的类别预测引导

# 两种核心特征
# query（查询 tokens）：来自 SideAdapterNetwork 的 Transformer 编码输出，用于定位全景图中的分割区域；
# x（视觉特征）：来自 DAO 处理后的畸变适配特征，包含全景图的空间与语义信息；

# 输出：对应论文 DAN 的双输出设计（3.2 节模型架构图）：
    # mask proposals：像素级掩码，每个查询对应一个全景图的分割区域（如 “行人”“道路” 的掩码）；
    # attention bias：注意力偏置，引导后续RecWithAttnbias模块（类别预测）聚焦关键区域，提升开放词汇分割精度。
    """Module for decoding query and visual features with MLP layers to
    generate the attention biases and the mask proposals."""

    def __init__(
        self,
        *,
        in_channels: int,          # 输入特征通道数（与SideAdapterNetwork的embed_dims一致，论文默认240）
        total_heads: int = 1,      # 注意力头数（与后续RecWithAttnbias的num_heads匹配，论文默认12）
        total_layers: int = 1,     # 注意力层数（与后续RecWithAttnbias的num_layers匹配，论文默认3）
        embed_channels: int = 256, # MLP输出的嵌入维度（统一 query和 x（2D 特征）的维度，为后续掩码生成的 “通道点积” 做准备）
        mlp_channels: int = 256,   # MLP隐藏层通道数（论文采用中等维度平衡性能与计算量）
        mlp_num_layers: int = 3,   # MLP层数（3层MLP足够捕捉复杂特征映射，论文实验验证）
        rescale_attn_bias: bool = False,  # 是否缩放注意力偏置（适配不同场景的偏置强度）
    ):
        # 初始化父类与参数保存：标准 PyTorch 模块初始化，保存关键参数用于后续 forward 计算。        
        super().__init__()
        self.total_heads = total_heads  # 保存注意力头数（后续生成对应数量的偏置）
        self.total_layers = total_layers  # 保存注意力层数（后续拆分偏置到每层）

        # 用partial包装nn.Conv2d，固定kernel_size=1——1×1 卷积可在不改变空间维度（H×W） 的前提下调整通道数，
        # 完美适配视觉特征x（[B,C,H,W]）的处理需求（论文需保留全景图的空间结构用于掩码生成）。
# partial是 Pythonfunctools模块提供的偏函数工具，
# 核心作用是：“冻结” 一个函数的部分参数，生成一个新的函数，调用新函数时，无需重复传入已冻结的参数，只需补充未指定的参数即可。
# 举个具体例子：
# 用原函数定义 1×1 卷积：nn.Conv2d(in_channels=240, out_channels=256, kernel_size=1)；
# 用新函数定义：dense_affine_func(in_channels=240, out_channels=256)（效果完全一致，代码更简洁）。kernel_size会自动设为 1，无需重复写。

# 为什么固定kernel_size=1？（1×1 卷积的核心特性）
    # 利用1×1 卷积的独特特性——在不改变特征图空间维度（H×W）的前提下，灵活调整通道数

    # 1×1 卷积的空间维度保留原理，即输入输出的空间维度完全一致
    # 1×1 卷积的通道调整能力，即通道维度的线性映射，本质：相当于对每个空间位置（h,w）的 C 维特征做一次 “全连接层” 操作，但不破坏空间结构。
        # 这正好解决论文中的 “维度统一” 需求：
        # query（[B,N,C]）通过普通 MLP 投影到 256 维；
        # x（[B,240,H,W]）通过 1×1 卷积投影到 256 维；
        # 两者通道维度一致后，才能通过torch.einsum做 “通道点积” 生成掩码（对应论文 3.2 节掩码生成逻辑）。

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)

        # Query Branch：处理查询tokens（1D序列，无空间维度）
            # 查询分支 MLP：
            # 输入：query（[B,N,C]，N 为查询数，论文默认 100）；
            # 功能：将查询 tokens 从in_channels投影到embed_channels，增强查询的语义表达能力（对应论文中 “查询需捕捉分割区域语义” 的设计）；
            # 为何用普通 MLP：查询是 1D 序列（无 H×W），无需保持空间维度，普通 Linear 层更高效。
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels,
                             mlp_num_layers)
        
        # Pixel Branch：处理视觉特征（2D特征，需保留空间维度）
            # 像素分支 MLP（论文核心设计之一）：
            # 输入：x（[B,C,H,W]，DAO 处理后的畸变适配特征）；
            # 关键差异：通过affine_func=dense_affine_func使用 1×1 卷积，确保处理后特征仍为 [B,embed_channels,H,W]，保留全景图的空间细节（如边缘畸变区域的像素位置）；
            # 作用：将视觉特征的通道维度从in_channels统一到embed_channels，与查询维度对齐，为后续掩码生成的 “通道点积” 做准备。
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,  # 用1×1卷积替代Linear，保留H×W
        )

        # Attention Bias Branch：生成注意力偏置（需适配头数和层数）
            # 注意力偏置分支 MLP（对应论文 “引导类别预测聚焦关键区域”）：
            # 输出维度设计：embed_channels × total_heads × total_layers—— 
                # 需为每一层 Transformer、每一个注意力头生成独立的空间偏置，确保偏置能精准引导不同层 / 头的注意力分布；
            # 空间保留：同样用 1×1 卷积，确保偏置的空间维度（H×W）与视觉特征一致，后续可与查询结合生成 “查询 - 空间” 关联的偏置。
        self.attn_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels * self.total_heads * self.total_layers,
            mlp_num_layers,
            affine_func=dense_affine_func,  # 1×1卷积保留空间维度，偏置需与视觉特征空间对齐
        )

        # 偏置缩放控制：
            # 可选线性层缩放，用于调整偏置的整体强度（如全景图畸变严重时可增大偏置权重）；
            # 默认恒等映射，论文实验验证无需额外缩放即可满足需求，避免引入多余参数。
        if rescale_attn_bias:
            self.bias_scaling = nn.Linear(1, 1)  # 线性层缩放偏置强度（适配不同畸变场景）
        else:
            self.bias_scaling = nn.Identity()  # 恒等映射（不缩放，论文默认配置）

    def forward(self, query: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward function.
        Args:
            query (Tensor): Query Tokens [B,N,C]  # B=批次，N=查询数（100），C=in_channels（240）
            x (Tensor): Visual features [B,C,H,W] # B=批次，C=in_channels（240），H/W=全景特征图尺寸
        Return:
            mask_preds (Tensor): Mask proposals [B,N,H,W] # 每个查询对应一个H×W的分割掩码
            attn_bias (List[Tensor]): List of attention bias # 每层一个偏置，形状[B,heads,N,H,W]

        输入输出定义：明确与前后模块的接口，
        query来自 SideAdapterNetwork 的 Transformer 查询输出，
        x来自 DAO 处理后的视觉特征；输出直接对接后续RecWithAttnbias（attn_bias）和分割结果生成（mask_preds）。
        """

        # 特征投影：
            # 将query和x分别通过各自的 MLP 投影到embed_channels（256），确保两者通道维度一致，
            # 为后续 “查询 - 像素” 关联计算奠定基础（论文核心步骤：统一特征维度）。
        query = self.query_mlp(query)  # [B,N,C] → [B,N,embed_channels]（240→256）
        pix = self.pix_mlp(x)          # [B,C,H,W] → [B,embed_channels,H,W]（240→256）

        # 维度信息提取：获取pix的关键维度，用于后续掩码和偏置的形状调整。
        b, c, h, w = pix.shape  # 提取批次（b）、嵌入通道（c=256）、空间维度（h/w）


        # preidict mask：生成掩码提案（对应论文3.2节“DAN输出掩码提案”）
            # 掩码提案生成（论文核心公式对应）：torch.einsum含义：按 “通道维度（c）” 对query（[B,N,c]）和pix（[B,c,H,W]）做点积，
            # 论文逻辑：每个查询（n）通过与视觉特征的通道点积，生成一个 H×W 的掩码 —— 查询捕捉 “分割区域语义”，视觉特征提供 “空间位置信息”，
            # 两者结合实现 “语义 - 空间” 对齐的掩码定位（如 “查询 1 对应行人掩码，查询 2 对应道路掩码”）。
        mask_preds = torch.einsum('bqc,bchw->bqhw', query, pix)


        # generate attn bias：生成注意力偏置（对应论文“引导类别预测聚焦”）
            # 初步生成偏置特征：通过attn_mlp处理视觉特征x，得到包含 “通道 - 头 - 层” 信息的偏置特征，后续需拆分与查询结合。
        attn = self.attn_mlp(x)  # [B,C,H,W] → [B, c×heads×layers, H,W]（240→256×12×3）
        # 偏置维度重组：
        # 将attn从 [B, c×heads×layers, H,W] 拆分为 [B, layers, heads, c, H, W]，明确区分 “层、头、通道、空间” 维度，为后续与查询结合做准备。
        attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)

        # 查询 - 偏置结合（论文核心关联步骤）：einsum含义：按 “通道维度（c）” 对query（[B,N,c]）和attn（[B,l,n,c,H,W]）做点积，
            # 论文逻辑：生成 “批次（b）- 层（l）- 头（n）- 查询（q）- 空间（h,w）” 的五维偏置
            #  —— 每个查询（对应一个掩码）在每个层 / 头都有独立的空间偏置，
            # 引导类别预测模块（RecWithAttnbias）在对应空间区域增强注意力（如掩码覆盖的 “行人区域” 偏置增大，聚焦该区域类别判断）。
        attn_bias = torch.einsum('bqc,blnchw->blnqhw', query, attn)

        # 偏置缩放与维度调整：
            # attn_bias[..., None]：添加最后一个维度（shape 变为 [B,l,n,q,h,w,1]），适配线性层输入；
            # 缩放后squeeze(-1)：移除多余维度，恢复为 [B,l,n,q,h,w]，确保偏置维度不变。
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)

        # 偏置分层处理：
            # chunk按dim=1（层维度）拆分，得到与total_layers数量一致的偏置列表；
            # 移除多余的层维度（1），最终每个偏置元素形状为 [B, heads, N, H, W]，可直接输入后续对应层的 Transformer，符合论文 “分层引导注意力” 的设计。
        attn_bias = attn_bias.chunk(self.total_layers, dim=1)  # 按层拆分，得到layers个[B,1,n,q,h,w]
        attn_bias = [attn.squeeze(1) for attn in attn_bias]   # 移除层维度，每个元素为[B,n,q,h,w]

        # 输出返回：
            # mask_preds：[B,N,H,W]，用于后续分割结果生成（与类别得分结合得到最终掩码）；
            # attn_bias：列表，每层一个 [B,heads,N,H,W] 的偏置，用于引导RecWithAttnbias的类别预测，两者共同构成论文 DAN 模块的核心输出。        
        return mask_preds, attn_bias


class SideAdapterNetwork(nn.Module):
    # 类内注释：明确CLIP特征融合的两种方式（论文适配不同畸变场景）
    # conv：1×1 卷积降维融合（轻量级，论文默认，平衡速度与性能）；
    # ca：CrossAttention 融合（精准跨域对齐，适用于畸变严重的全景图，如WildPASS户外场景）；
    # 当fuse_type='ca'时，初始化CrossAttention实例列表，在指定Transformer层融合CLIP特征
    """Side Adapter Network for predicting mask proposals and attention bias.
    Args:
        in_channels (int): 输入图像通道数（默认RGB=3，与论文一致）. Default: 3.
        clip_channels (int): CLIP视觉特征维度（ViT-B/16为768，论文固定）. Default: 768.
        embed_dims (int): 侧网络嵌入维度（论文设240，平衡计算量与语义表达）. Default: 240.
        patch_size (int): 图像分块大小（与CLIP ViT-B/16一致为16，确保特征粒度匹配）. Default: 16.
        patch_bias (bool): PatchEmbed是否加偏置（默认True，提升训练稳定性）. Default: True.
        num_queries (int): 掩码提案数量（论文设100，覆盖全景图常见目标数量）. Default: 100.
        fusion_index (List[int]): 融合CLIP的Transformer层索引（默认[0,1,2,3]，对应CLIP 4个中间层）. Default: [0, 1, 2, 3].
            补充逻辑：每层融合对应CLIP的一个中间层特征，流程为：
                1. 格式转换：CLIP 4D特征（N,C,H,W）→ 3D序列（N,H×W,C）（适配CrossAttention输入）；
                2. 跨注意力计算：侧网络特征为Q，CLIP特征为K/V，捕捉语义关联；
                3. 特征替换：融合后特征替换侧网络原视觉token，用于后续DAO和Transformer编码；
        cfg_encoder (ConfigType): Transformer编码器配置（层数、头数等，论文设4层编码器）.
        cfg_decoder (ConfigType): MLPMaskDecoder配置（头数、层数等，与后续RecWithAttnbias匹配）.
        norm_cfg (dict): 归一化配置（默认LayerNorm，与CLIP保持一致）. Default: dict(type='LN').
    """
    # 2. __init__方法：初始化 DAN 核心组件（论文模块映射）
    def __init__(
            self,
            in_channels: int = 3,          # 输入图像通道（RGB=3）
            clip_channels: int = 768,      # CLIP特征维度（768）
            embed_dims: int = 240,         # 侧网络嵌入维度（240）
            patch_size: int = 16,          # Patch大小（16×16） 代码默认值，即“每个图像块的边长为16像素”
            patch_bias: bool = True,       # PatchEmbed偏置（True）
            num_queries: int = 100,        # 掩码查询数（100）
            fusion_index: list = [0, 1, 2, 3],  # CLIP融合层索引（4层）
            fuse_type: str = 'conv',       # 融合方式（conv/ca）
            cfg_encoder: ConfigType = ..., # Transformer编码器配置
            cfg_decoder: ConfigType = ..., # MLPMaskDecoder配置
            norm_cfg: dict = dict(type='LN'),  # 归一化（LN）
            dao_after_fusion: bool = False,    # 融合后是否用DAO（False默认）
            cfg_dao: ConfigType = ...,     # DAO模块配置
    ):
        super().__init__()  # 初始化父类nn.Module

    # 2.1 初始化 DAO 模块（处理全景畸变，论文核心创新）
        # 1. 初始化“融合前DAO”列表（按fusion_index长度，每层融合对应一个DAO）
            # 论文逻辑：DAO 是处理全景畸变的核心，每个 CLIP 融合层前都需用 DAO 适配畸变，
            #   因此按fusion_index长度初始化（默认 4 个 DAO）。
        dao_list = []
        for i in range(len(fusion_index)):
            dao_list.append(
                DAO(  # 调用DAO类（../dcnv3/modules/dao.py），参数来自cfg_dao
                    channels=cfg_dao.channels,          # 输入通道（240，与embed_dims一致）
                    group=cfg_dao.group,                # DAO分组数（论文参考DCNv3设8）
                    kernel_size=cfg_dao.kernel_size,    # 卷积核大小（3×3，捕捉局部畸变）
                    stride=cfg_dao.stride,              # 步长（1，不改变尺寸）
                    pad=cfg_dao.pad,                    # Padding（1，保证尺寸不变）
                    dilation=cfg_dao.dilation,          # 膨胀率（1）
                    offset_scale=cfg_dao.offset_scale,  # 偏移缩放（1.0，控制畸变适配强度）
                    act_layer=cfg_dao.act_layer,        # 激活函数（GELU，与CLIP一致）
                    norm_layer=cfg_dao.norm_layer,      # 归一化（LN）
                    dw_kernel_size=cfg_dao.dw_kernel_size,  # 深度卷积核（5×5）
                    center_feature_scale=cfg_dao.center_feature_scale,  # 中心特征缩放（True）
                    pa_kernel_size=cfg_dao.pa_kernel_size,  # PatchAttention核（3×3）
                    pa_norm_layer=cfg_dao.pa_norm_layer,    # PatchAttention归一化（Softmax）
                ))
        self.dao_list = nn.ModuleList(dao_list)  # 包装为ModuleList（确保参数可训练）

        # 2. 可选：初始化“融合后DAO”列表（若dao_after_fusion=True，融合后再处理一次畸变）
            # 设计目的：针对畸变严重的场景（如全景图边缘），融合 CLIP 特征后可能引入新的特征分布差异，
            #   额外 DAO 处理可进一步增强畸变适配能力（论文实验中默认关闭，按需开启）。
        self.dao_after_fusion = dao_after_fusion
        if dao_after_fusion:
            dao_after_fusion_list = []
            for i in range(len(fusion_index)):
                dao_after_fusion_list.append(
                    DAO(
                        channels=cfg_dao.channels,
                        group=cfg_dao.group,
                        kernel_size=cfg_dao.kernel_size,
                        stride=cfg_dao.stride,
                        pad=cfg_dao.pad,
                        dilation=cfg_dao.dilation,
                        offset_scale=cfg_dao.offset_scale,
                        act_layer=cfg_dao.act_layer,
                        norm_layer=cfg_dao.norm_layer,
                        dw_kernel_size=cfg_dao.dw_kernel_size,
                        center_feature_scale=cfg_dao.center_feature_scale,
                        pa_kernel_size=cfg_dao.pa_kernel_size,
                        pa_norm_layer=cfg_dao.pa_norm_layer,
                    ))
            self.dao_after_fusion_list = nn.ModuleList(dao_after_fusion_list)
            
    # 2.2 初始化 PatchEmbed（图像分块，对应 ViT 的 Patch 嵌入）
        # 核心作用：将输入图像（[B,3,640,640]）转为特征序列（[B, 1600, 240]），计算过程：
            # 分块数量：(640/16)×(640/16) = 40×40 = 1600 个 patch；
            # 维度转换：[B,3,640,640] → [B,240,40,40] → [B,1600,240]；
            # 论文适配：与 CLIP 的 Patch 嵌入逻辑一致，确保后续融合时特征粒度匹配（CLIP ViT-B/16 也是 16×16 patch）。
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,          # 输入通道（3）
            embed_dims=embed_dims,            # 嵌入维度（240）
            conv_type='Conv2d',               # 卷积类型（2D卷积）
            kernel_size=patch_size,           # 核大小（16×16，与patch_size一致） 代码默认值，即“每个图像块的边长为16像素”
            stride=patch_size,                # 步长（16，无重叠分块）
            padding=0,                        # Padding（0，输入尺寸640×640正好被16整除）
            input_size=(640, 640),            # 输入图像尺寸（论文训练默认640×640）
            bias=patch_bias,                  # 是否加偏置（True）
            norm_cfg=None,                    # 不归一化（后续与CLIP融合后处理）
            init_cfg=None,
        )

        # 提取PatchEmbed的输出尺寸（ori_h=40, ori_w=40）和patch总数（1600）
        ori_h, ori_w = self.patch_embed.init_out_size
        num_patches = ori_h * ori_w  # 40×40=1600


    # 2.3 初始化位置嵌入与查询嵌入（生成 mask 提案的基础）

        # 1. 图像patch的位置嵌入（捕捉patch的空间位置信息）
            # 论文逻辑：Transformer 无空间归纳偏置，位置嵌入为 patch 添加 “位置信息”
            #   （如全景图中 “左上” 与 “右下” patch 的位置差异），确保后续能生成空间正确的 mask 提案。
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dims) * .02)  # [1,1600,240]，std=0.02初始化
        # 2. 查询token的位置嵌入（为mask提案查询添加位置信息）
            # 设计目的：100 个查询 token 对应 100 个 mask 提案，位置嵌入帮助查询捕捉 “不同 mask 的空间区分度”
            #   （如 “行人” mask 与 “道路” mask 的位置差异）。
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))  # [1,100,240]，初始化为0
        # 3. 查询token的可学习嵌入（初始化mask提案的语义查询）
            # 论文对应：查询 token 是生成 mask 提案的 “语义种子”，
            #   训练中学习不同类别 mask 的语义特征（如 “行人” 查询 token 学习行人的语义模式）。
        self.query_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))  # [1,100,240]，初始化为0
        
    # 2.4 初始化 Transformer 编码器层（提取高级视觉特征）
        # 核心作用：对 “查询 token + 图像 patch” 的混合序列进行编码，捕捉全局语义关联
        #   （如全景图中 “行人” 与 “道路” 的语义依赖），为后续 mask 提案生成提供高级特征。    
        encode_layers = []
        for i in range(cfg_encoder.num_encode_layer):  # cfg_encoder.num_encode_layer默认4层
            encode_layers.append(
                TransformerEncoderLayer(  # MMSeg的Transformer编码器层（与ViT一致）
                    embed_dims=embed_dims,                  # 嵌入维度（240）
                    num_heads=cfg_encoder.num_heads,        # 注意力头数（默认8，240/8=30为头维度）
                    feedforward_channels=cfg_encoder.mlp_ratio * embed_dims,  # MLP隐藏层（4×240=960）
                    norm_cfg=norm_cfg))                     # 归一化（LN）
        self.encode_layers = nn.ModuleList(encode_layers)  # 包装为ModuleList

    # 2.5 初始化 CLIP 融合模块（conv 或 ca 方式）
        # 论文逻辑：CLIP 特征是 768 维，侧网络是 240 维，融合模块需解决 “维度匹配” 和 “跨域对齐”：
        # conv 方式：轻量快速，适合资源有限场景；（论文默认）
        # ca 方式：通过跨注意力捕捉 CLIP 与侧网络的语义关联，适合畸变严重、需精准对齐的场景。

        self.fuse_type = fuse_type  # 保存融合方式（conv/ca）

        # 方式1：conv融合（1×1卷积降维，轻量级，论文默认）
        if fuse_type == 'conv':
            conv_clips = []
            for i in range(len(fusion_index)):  # 按融合层数初始化（4个）
                conv_clips.append(
                    nn.Sequential(
                        LayerNorm2d(clip_channels),  # CLIP特征归一化（保持空间维度）
                        ConvModule(  # 1×1卷积：将CLIP的768维→240维（与侧网络维度一致）
                            clip_channels,          # 输入通道（768）
                            embed_dims,             # 输出通道（240）
                            kernel_size=1,          # 1×1卷积（不改变空间维度）
                            norm_cfg=None,          # 不归一化
                            act_cfg=None)))         # 无激活（线性融合）
            self.conv_clips = nn.ModuleList(conv_clips)  # 包装为ModuleList

        # 方式2：ca融合（CrossAttention，精准跨域对齐）
        elif fuse_type == 'ca':
            ca_clips = []
            for i in range(len(fusion_index)):  # 按融合层数初始化（4个）
                ca_clips.append(
                    CrossAttention(  # 调用CrossAttention类（ops_head.py）
                        dim=clip_channels,          # CLIP特征维度（768）
                        num_heads=16,               # 注意力头数（16，768/16=48为头维度）
                        qkv_bias=False,             # 无QKV偏置
                        qk_scale=None,              # 缩放因子（默认头维度开方倒数）
                        attn_drop=0.,               # 注意力Dropout（0.0）
                        proj_drop=0.,               # 输出Dropout（0.0）
                        attn_head_dim=None,         # 头维度默认（768/16=48）
                        out_dim=embed_dims))        # 输出维度（240，匹配侧网络）
            self.ca_clips = nn.ModuleList(ca_clips)  # 包装为ModuleList

        # 不支持其他融合方式
        else:
            raise NotImplementedError(f'融合方式{fuse_type}未实现，仅支持conv/ca')


    # 2.6 初始化 MLPMaskDecoder（生成 mask 提案与注意力偏置）
        # 论文对应：接收encode_feature输出的 “查询 + 视觉特征”，
        #   生成最终的 mask 提案（[B,100,H,W]）和注意力偏置（每层 [B,12,100,H,W]），
        #   直接对接后续RecWithAttnbias模块。
        self.fusion_index = fusion_index  # 保存CLIP融合层索引
        self.mask_decoder = MLPMaskDecoder(  # 调用MLPMaskDecoder类（ops_head.py）
            in_channels=embed_dims,                # 输入通道（240）
            total_heads=cfg_decoder.num_heads,     # 注意力头数（与RecWithAttnbias一致，12）
            total_layers=cfg_decoder.num_layers,   # 注意力层数（与RecWithAttnbias一致，3）
            embed_channels=cfg_decoder.embed_channels,  # MLP嵌入维度（256）
            mlp_channels=cfg_decoder.mlp_channels,      # MLP隐藏层（256）
            mlp_num_layers=cfg_decoder.num_mlp,        # MLP层数（3）
            rescale_attn_bias=cfg_decoder.rescale)     # 偏置是否缩放（False）
        
    # 3. init_weights方法：权重初始化（确保训练稳定）
        # 初始化逻辑：遵循 Transformer 和卷积层的最佳实践，避免训练初期梯度消失 / 爆炸：
            # 位置嵌入用 trunc_normal（避免极端值）；
            # 线性层 / 卷积层用 xavier（保证输入输出方差一致）。
    def init_weights(self):
        # 1. 图像位置嵌入：截断正态分布初始化（std=0.02，ViT标准初始化）
        trunc_normal_(self.pos_embed, std=0.02)
        # 2. 查询嵌入与查询位置嵌入：正态分布初始化（std=0.02）
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        # 3. 若为conv融合：conv_clips中的1×1卷积用caffe2_xavier初始化（适合卷积层）
        if self.fuse_type == 'conv':
            for i in range(len(self.conv_clips)):
                caffe2_xavier_init(self.conv_clips[i][1].conv)  # conv_clips[0][1]是ConvModule

    # 4. CLIP 融合方法（conv/ca 两种方式）

    # 4.1 fuse_clip：conv 方式融合（轻量级）
        # 论文逻辑：conv 方式通过 “线性降维 + 特征相加” 实现轻量级融合，适合全景图尺寸较大、需控制计算量的场景。
    def fuse_clip(self, fused_index: int, x: torch.Tensor, clip_feature: torch.Tensor, hwshape: Tuple[int, int], L: int):
        """Fuse CLIP feature and visual tokens（conv方式：1×1卷积+特征相加）"""
        # 1. CLIP特征处理：归一化→1×1卷积降维→resize到侧网络特征尺寸
        fused_clip = (resize(
            self.conv_clips[fused_index](clip_feature.contiguous()),  # [B,768,Hc,Wc]→[B,240,Hc,Wc]
            size=hwshape,  # resize到侧网络特征尺寸（如40×40）
            mode='bilinear',  # 双线性插值（平滑缩放）
            align_corners=False)).permute(0, 2, 3, 1).reshape(x[:, -L:, ...].shape)
        # 维度转换说明：
        # resize后：[B,240,H,W] → permute→[B,H,W,240] → reshape→[B,L,240]（L=H×W=1600）

        # 2. 特征融合：侧网络的图像特征（x[:, -L:, ...]） + CLIP融合特征（fused_clip）
        x = torch.cat([x[:, :-L, ...], x[:, -L:, ...] + fused_clip], dim=1)
            # x结构说明：x[:, :-L, ...]是查询token（[B,100,240]），x[:, -L:, ...]是图像特征（[B,1600,240]）
                # 融合后图像特征变为：原图像特征 + CLIP特征（保留原特征的同时注入CLIP知识）
        
        return x  # 返回融合后的x：[B,100+1600,240]

    # 4.2 fuse_clip_ca：ca 方式融合（精准跨域对齐）
        # 论文优势：相比 conv 的 “全局相加”，ca 方式通过注意力权重聚焦 CLIP 中与侧网络特征相关的区域
        #   （如畸变区域的目标特征），对齐精度更高，适合畸变严重的全景场景。

    def fuse_clip_ca(self, fused_index: int, x: torch.Tensor, clip_feature: torch.Tensor, hwshape: Tuple[int, int], L: int):
        # CLIP特征格式转换：4D（N,C,H,W）→ 3D序列（N,H×W,C）（适配CrossAttention输入）
        k = rearrange(clip_feature, 'N C H W -> N (H W) C').contiguous()  # [B, Hc×Wc, 768]
        v = rearrange(clip_feature, 'N C H W -> N (H W) C').contiguous()  # K/V均为CLIP特征（跨注意力中K=V）
            # 当 (K=V) 时，意味着 “信息的索引” 和 “信息的内容” 完全一致 
                # ——CLIP 特征既是 “语义索引”（K），也是 “语义内容”（V）。
                # 这是因为 CLIP 特征本身已经是 “语义 - 视觉对齐” 的高质量特征
                # （预训练时通过图像 - 文本对学习，包含 “行人”“道路” 等通用语义），无需额外分离索引和内容。
        # 调用CrossAttention融合：侧网络特征x为Q，CLIP特征为K/V
        fused_clip = self.ca_clips[fused_index](x, k, v)  # [B,100+1600,240]
        # 融合逻辑：通过跨注意力，x（侧网络特征）学习CLIP特征中的通用语义（如“行人”“道路”的视觉模式）

            # 侧网络特征 x：是经过 DAO 模块处理的 “任务相关特征”，专注于全景图像的畸变校正和局部细节
                # （比如弯曲的道路边缘、畸变的建筑轮廓），但缺乏 “通用类别语义”（不知道 “这是行人”“那是天空”）。
                # 作为 Q，它的作用是主动 “查询”：“我（侧网络）需要从 CLIP 中获取哪些语义信息来解释我看到的局部细节？”

            # CLIP 特征：是经过大规模预训练的 “通用语义特征”，包含丰富的开放词汇知识
                # （通过图像 - 文本对学习，能匹配 “行人”“道路” 等文本描述），但可能不适应全景图像的畸变。
                # 作为 K/V，它的作用是被动 “提供”：“我（CLIP）有这些通用语义，你（侧网络）需要什么就拿什么。”

        # 跨注意力为什么能让 x 学习 CLIP 的通用语义？
            # 跨注意力的核心能力是 “基于相关性的信息筛选与融合”
            # 具体过程如下：
                # 1.计算相关性：侧网络特征 x（Q）与 CLIP 特征（K）计算相似度（\(QK^T\)），
                    # 得到 “每个侧网络特征 token 需要多少 CLIP 语义” 的权重
                    # （比如 x 中 “某个畸变区域的 token” 与 CLIP 中 “行人语义 token” 的相似度高）。
                # 2.加权融合：用上述权重对 CLIP 特征（V）进行加权求和，得到 “CLIP 中与侧网络特征最相关的语义信息”
                    # （比如将 “行人” 语义赋予 x 中的对应区域）。
                # 3.更新特征：将融合结果输出给 x，使 x 既保留自身的畸变校正能力，又带上了 CLIP 的通用语义
                    # （现在 x 知道 “这个畸变区域是行人”）。
                    # 简言之，跨注意力通过 “Q 引导方向、K 计算相关性、V 提供内容” 的机制，
                    # 让侧网络特征 x 主动 “抓取” CLIP 中最相关的通用语义，实现了 “任务特征” 与 “通用知识” 的精准结合。
        
        return fused_clip  # 返回融合后的侧网络特征


    # 5. encode_feature：核心编码流程（图像→DAO→CLIP 融合→Transformer）
    # 核心流程总结（对应论文 3.2 节模型架构图）：
        # 图像→PatchEmbed→位置嵌入拼接→DAO 处理畸变→CLIP 融合→Transformer 编码→提取监督特征，
        #   每一步均围绕 “适配全景畸变” 和 “融合 CLIP 知识” 展开，最终为解码模块提供高质量特征。
    def encode_feature(self, image: torch.Tensor, clip_features: List[torch.Tensor], deep_supervision_idxs: List[int]) -> List[List]:
        """Encode images by a lightweight vision transformer（论文核心编码流程）"""
        # 校验：融合层数需与CLIP特征层数一致（fusion_index长度=clip_features长度）
        assert len(self.fusion_index) == len(clip_features)

        # 1. 图像分块嵌入：image→x（特征序列）+ hwshape（特征图尺寸）
        x, hwshape = self.patch_embed(image)  # x: [B,1600,240]; hwshape: (40,40) 即分块数量（640/16=40）
        ori_h, ori_w = self.patch_embed.init_out_size  # 原始patch尺寸（40,40）
        pos_embed = self.pos_embed  # 初始位置嵌入：[1,1600,240]

        # 2. 位置嵌入自适应调整（若图像尺寸变化，resize位置嵌入）
        if self.pos_embed.shape[1] != x.shape[1]:
            pos_embed = (
                resize(
                    self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),  # [1,1600,240]→[1,240,40,40]
                    size=hwshape,  # resize到当前hwshape（如输入不是640×640时）
                    mode='bicubic',  # 双三次插值（比双线性更精准）
                    align_corners=False,
                ).flatten(2).permute(0, 2, 1))  # [1,240,H,W]→[1,H×W,240]（恢复序列格式）

        # 3. 拼接查询嵌入与位置嵌入（构建“查询+图像”混合序列）
        # DETR架构
        # 位置嵌入拼接：查询位置嵌入（[1,100,240]） + 图像位置嵌入（[1,1600,240]）→ [1,1700,240]
        pos_embed = torch.cat([
            self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed
        ], dim=1)
        # 特征序列拼接：查询嵌入（[1,100,240]） + 图像特征（[B,1600,240]）→ [B,1700,240]
        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)
        # 添加位置嵌入：x = 特征序列 + 位置嵌入（注入空间信息）
        x = x + pos_embed

        # 4. 初始化关键变量：L=图像patch数量（40×40=1600），fused_index=CLIP融合索引（从0开始）
        L = hwshape[0] * hwshape[1]
        fused_index = 0

        # 5. 处理第0层融合（若fusion_index[0] == 0，在第1个Transformer层前融合）
        if self.fusion_index[fused_index] == 0:
            # 5.1 DAO处理图像特征：序列→空间格式→DAO→序列格式
            x1 = rearrange(x[:, -L:, ...], 'N (H W) C -> N H W C', H=hwshape[0])  # [B,1600,240]→[B,40,40,240]（DAO输入格式）
            x1 = self.dao_list[fused_index](x1)  # DAO处理畸变（输出[B,40,40,240]）
            x1 = rearrange(x1, 'N H W C -> N (H W) C')  # [B,40,40,240]→[B,1600,240]（恢复序列格式）
            x = torch.cat([x[:, :-L, ...], x1], dim=1)  # 替换原图像特征（注入DAO畸变处理结果）

            # 5.2 CLIP融合（conv或ca方式）
            if self.fuse_type == 'conv':
                x = self.fuse_clip(fused_index, x, clip_features[0][0], hwshape, L)
            elif self.fuse_type == 'ca':
                x = self.fuse_clip_ca(fused_index, x, clip_features[0][0], hwshape, L)

            # 5.3 可选：融合后DAO处理
            if self.dao_after_fusion:
                x1 = rearrange(x[:, -L:, ...], 'N (H W) C -> N H W C', H=hwshape[0])
                x1 = self.dao_after_fusion_list[fused_index](x1)
                x1 = rearrange(x1, 'N H W C -> N (H W) C')
                x = torch.cat([x[:, :-L, ...], x1], dim=1)

            fused_index += 1  # 融合索引+1（处理下一层）

        # 6. 遍历Transformer编码器层，处理后续融合与编码
        outs = []  # 保存深度监督的特征（query + x_feat）
        for index, block in enumerate(self.encode_layers, start=1):  # index从1开始（对应Transformer层1~4）
            # 6.1 Transformer编码（捕捉全局语义关联）
            x = block(x)  # [B,1700,240]→[B,1700,240]

            # 6.2 处理后续融合（若当前层是fusion_index中的索引）
            if index < len(self.fusion_index) and index == self.fusion_index[fused_index]:
                # DAO处理图像特征（逻辑与第0层一致）
                x1 = rearrange(x[:, -L:, ...], 'N (H W) C -> N H W C', H=hwshape[0])
                x1 = self.dao_list[fused_index](x1)
                x1 = rearrange(x1, 'N H W C -> N (H W) C')
                x = torch.cat([x[:, :-L, ...], x1], dim=1)

                # CLIP融合（conv或ca方式）
                if self.fuse_type == 'conv':
                    x = self.fuse_clip(fused_index, x, clip_features[fused_index][0], hwshape, L)
                elif self.fuse_type == 'ca':
                    x = self.fuse_clip_ca(fused_index, x, clip_features[fused_index][0], hwshape, L)

                # 融合后DAO处理（可选）
                if self.dao_after_fusion:
                    x1 = rearrange(x[:, -L:, ...], 'N (H W) C -> N H W C', H=hwshape[0])
                    x1 = self.dao_after_fusion_list[fused_index](x1)
                    x1 = rearrange(x1, 'N H W C -> N (H W) C')
                    x = torch.cat([x[:, :-L, ...], x1], dim=1)

                fused_index += 1  # 融合索引+1

            # 6.3 提取查询特征（x_query）和视觉特征（x_feat）
                # 从融合特征中分离 “查询” 与 “视觉”
                # 取序列的前 100 个 token，对应初始化时拼接的 “查询嵌入”，负责引导后续 MLPMaskDecoder 生成 100 个掩码提案（每个查询对应一个提案）。
            x_query = x[:, :-L, ...]  # [B,100,240]（查询token，用于生成mask提案）
                # 视觉特征转换：[B,1600,240]→[B,240,1600]→[B,240,40,40]（空间格式，用于MLPMaskDecoder）
            x_feat = x[:, -L:, ...].permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], hwshape[0], hwshape[1])

            # 6.4 深度监督：保存当前层特征（若为监督层或最后一层）
            if index in deep_supervision_idxs or index == len(self.encode_layers):
                outs.append({'query': x_query, 'x': x_feat})  # 每个元素含“查询+视觉特征”

            # 6.5 非最后一层：添加位置嵌入（避免位置信息丢失）
            if index < len(self.encode_layers):
                x = x + pos_embed

        return outs  # 返回深度监督特征列表（用于后续MLPMaskDecoder解码）

    # 6. decode_feature：解码生成 mask 提案与注意力偏置
        # 论文对应：接收encode_feature的输出，生成论文中 DAN 的两大核心输出，直接对接后续模块：
            # mask_embeds→用于分割结果生成（与类别得分相乘）；
            # attn_biases→用于RecWithAttnbias引导类别预测。
    def decode_feature(self, features):
        mask_embeds = []  # 存储各监督层的mask提案
        attn_biases = []  # 存储各监督层的注意力偏置
        for feature in features:  # 遍历每个深度监督层的特征（outs中的元素）
            # 调用MLPMaskDecoder解码：feature含'query'和'x'，**feature表示解包传入
            mask_embed, attn_bias = self.mask_decoder(**feature)
            mask_embeds.append(mask_embed)  # mask_embed: [B,100,H,W]
            attn_biases.append(attn_bias)   # attn_bias: 列表，每层[B,12,100,H,W]
        return mask_embeds, attn_biases

    # 7. forward：前向传播入口（编码→解码）
        # 接口作用：作为侧网络与外部模块（如OPSCLIPHead）的对接入口，输入图像和 CLIP 特征，
        #   输出直接用于后续的类别预测和损失计算，完成论文中 DAN 模块的完整功能。
    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor], deep_supervision_idxs: List[int]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward function（侧网络完整前向流程）"""
        # 1. 特征编码：image→DAO→CLIP融合→Transformer→监督特征
        features = self.encode_feature(image, clip_features, deep_supervision_idxs)
        # 2. 特征解码：监督特征→mask提案+注意力偏置
        mask_embeds, attn_biases = self.decode_feature(features)
        return mask_embeds, attn_biases  # 返回给OPSCLIPHead（ops_head.py）




class RecWithAttnbias(nn.Module):
    """Mask recognition module by applying the attention biases to rest deeper
    CLIP layers. （将注意力偏置应用于CLIP深层网络，实现掩码类别的识别）

    Args:
        sos_token_format (str): 起始标记(sos)的格式，用于引导类别预测
            可选：["cls_token", "learnable_token", "pos_embedding"]
            Default: 'cls_token'.
        sos_token_num (int): sos标记的数量，需与查询数(num_queries)一致
            Default: 100.
        num_layers (int): 用于掩码识别的CLIP深层网络层数
            Default: 3.
        cross_attn (bool): 是否使用交叉注意力更新sos标记
            Default: False.
        embed_dims (int): CLIP层的特征维度（ViT-B/16为768）
            Default: 768.
        num_heads (int): 多头注意力的头数（CLIP ViT-B为12）
            Default: 12.
        mlp_ratio (int): MLP隐藏层维度与嵌入维度的比例
            Default: 4.
        qkv_bias (bool): 多头注意力中是否使用偏置
            Default: True.
        out_dims (int): 输出掩码提案的通道数，需与文本编码器输出维度一致
            Default: 512.
        final_norm (True): 是否对sos标记使用归一化层
        act_cfg (dict): FFN的激活函数配置
            Default: dict(type='GELU').
        norm_cfg (dict): 归一化层配置
            Default: dict(type='LN').
        frozen_exclude (List): 不冻结的参数列表（用于微调）

    主要作用是：
        利用注意力偏置引导类别预测：将MLPMaskDecoder生成的注意力偏置注入 CLIP 深层网络，使模型聚焦于掩码提案对应的区域，提升类别识别精度。
        灵活的 sos 标记设计：支持三种起始标记格式，适配不同场景的类别预测需求：
                cls_token：复用 CLIP 的类别标记，适合与预训练知识对齐；
                learnable_token：可学习标记，适合特定任务微调；
                pos_embedding：结合位置信息的标记，增强空间感知。
        双注意力模式：
                交叉注意力：sos 标记仅与图像特征交互，计算高效；
                自注意力：sos 标记与所有特征（包括自身和图像）交互，捕捉全局关联。
        最终输出的sos_token可与文本编码器生成的类别特征计算相似度，实现 “掩码 - 类别” 的匹配，完成开放词汇场景下的全景分割。    
    """

    def __init__(self,
                 sos_token_format: str = 'cls_token',
                 sos_token_num: int = 100,
                 num_layers: int = 3,
                 cross_attn: bool = False,
                 embed_dims: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 out_dims: int = 512,
                 final_norm: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 frozen_exclude: List = []):
        super().__init__()

        # 校验sos_token_format的合法性
        assert sos_token_format in [
            'cls_token', 'learnable_token', 'pos_embedding'
        ]
        self.sos_token_format = sos_token_format  # 保存sos标记格式
        self.sos_token_num = sos_token_num        # 保存sos标记数量
        self.frozen_exclude = frozen_exclude      # 保存不冻结参数列表
        self.cross_attn = cross_attn              # 保存是否使用交叉注意力
        self.num_layers = num_layers              # 保存Transformer层数
        self.num_heads = num_heads                # 保存注意力头数

        # 初始化sos标记（根据不同格式）
        if sos_token_format in ['learnable_token', 'pos_embedding']:
            # 可学习的sos标记：[sos_token_num, 1, embed_dims]，初始化为随机正态分布
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, embed_dims))  # 原代码中proj尚未定义，此处修正为embed_dims
            # 记录需冻结的参数（sos_token默认冻结，除非在frozen_exclude中）
            self.frozen = ['sos_token'] if not any('sos_token' in ex for ex in frozen_exclude) else []

        # 构建Transformer层（用于深层特征编码）
        layers = []
        for i in range(num_layers):
            layers.append(
                BaseTransformerLayer(  # 基础Transformer层（含自注意力和FFN）
                    attn_cfgs=dict(
                        type='MultiheadAttention',  # 多头自注意力
                        embed_dims=embed_dims,      # 嵌入维度（768）
                        num_heads=num_heads,        # 注意力头数（12）
                        batch_first=False,          # 输入格式为[序列长度, 批次, 维度]
                        bias=qkv_bias),             # 使用QKV偏置
                    ffn_cfgs=dict(
                        type='FFN',                 # 前馈网络
                        embed_dims=embed_dims,      # 嵌入维度（768）
                        feedforward_channels=mlp_ratio * embed_dims,  # MLP隐藏层维度（4×768=3072）
                        act_cfg=act_cfg),           # 激活函数（GELU）
                    # 操作顺序：归一化→自注意力→归一化→FFN
                    operation_order=('norm', 'self_attn', 'norm', 'ffn')))
        self.layers = nn.ModuleList(layers)  # 包装为ModuleList

        # 输出归一化层（LN）
        self.ln_post = build_norm_layer(norm_cfg, embed_dims)[1]
        # 投影层：将768维特征映射到out_dims（512），与文本编码器输出维度对齐
        self.proj = nn.Linear(embed_dims, out_dims, bias=False)

        self.final_norm = final_norm  # 是否最终归一化
        self._freeze()  # 冻结指定参数

    def init_weights(self, rec_state_dict):
        """初始化权重（支持加载预训练参数）"""
        # 初始化sos_token（若存在）
        if hasattr(self, 'sos_token'):
            normal_init(self.sos_token, std=0.02)  # 正态分布初始化，std=0.02
        # 加载预训练状态字典
        if rec_state_dict is not None:
            load_state_dict(self, rec_state_dict, strict=False, logger=None)
        else:
            super().init_weights()  # 调用父类初始化

    def _freeze(self):
        """冻结参数（除frozen_exclude列表中的参数外）"""
        if 'all' in self.frozen_exclude:  # 'all'表示不冻结任何参数
            return
        for name, param in self.named_parameters():
            # 若参数名不含任何不冻结关键字，则冻结
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _build_attn_biases(self, attn_biases, target_shape):
        """将注意力偏置转换为适合Transformer层的格式
        
        Args:
            attn_biases: 输入的注意力偏置列表，来自MLPMaskDecoder
            target_shape: 目标形状（H,W），与图像特征尺寸匹配
        Returns:
            formatted_attn_biases: 格式化后的注意力偏置列表
        """
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # attn_bias形状：[N, num_head, num_sos, H, W]
            # N=批次，num_head=注意力头数，num_sos=sos标记数，H/W=特征图尺寸
            n, num_head, num_sos, h, w = attn_bias.shape
            
            # 1. 重塑并下采样偏置，匹配目标形状（如原图特征尺寸）
            attn_bias = F.adaptive_max_pool2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),  # [N, num_head×num_sos, H, W]
                output_size=target_shape)  # 下采样到target_shape
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)  # 恢复为[N, num_head, num_sos, H, W]
            
            # 2. 调整注意力头数，与当前层的num_heads匹配
            true_num_head = self.num_heads
            assert (num_head == 1 or num_head == true_num_head), \
                f'输入偏置头数{num_head}与当前层{true_num_head}不兼容'
            if num_head == 1:
                # 若输入为单头，扩展到true_num_head
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            
            # 3. 重塑为[N×true_num_head, num_sos, L]，L=H×W（图像特征序列长度）
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]  # 图像特征序列长度
            
            # 4. 根据是否使用交叉注意力，生成不同格式的偏置
            if self.cross_attn:
                # 交叉注意力：偏置形状为[n×num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # 自注意力：构建完整的注意力掩码矩阵[n×num_head, total_len, total_len]
                # total_len = num_sos（sos标记） + 1（cls_token） + L（图像特征）
                total_len = num_sos + 1 + L
                new_attn_bias = attn_bias.new_zeros(total_len, total_len)
                # 初始化掩码：sos标记之间允许注意力，与其他区域的注意力需限制
                new_attn_bias[:, :num_sos] = -100  # 非sos区域对sos区域的注意力设为-100（抑制）
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0  # sos自身注意力设为0（允许）
                new_attn_bias[:num_sos, num_sos] = -100  # sos对cls_token的注意力设为-100（抑制）
                # 扩展到批次和头数维度
                new_attn_bias = new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                # 将图像特征区域的偏置填入：sos标记对图像特征的注意力
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)
        
        # 若输入偏置数量为1，复制到所有层
        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in range(self.num_layers)]
        return formatted_attn_biases

    def forward(self, bias: List[Tensor], feature: List[Tensor]):
        """前向传播：识别掩码的类别
        
        Args:
            bias (List[Tensor]): Transformer层的注意力偏置列表
            feature (List[Tensor]): 图像编码器的输出，包含cls_token和图像特征
        Returns:
            sos_token: 处理后的sos标记，用于与文本特征计算相似度（类别预测）
        """
        # 提取CLIP的cls_token和图像特征
        cls_token = feature[1].unsqueeze(0)  # [1, B, 768]（添加序列维度）
        img_feature = feature[0]             # [B, 768, H, W]（图像特征）
        b, c, h, w = img_feature.shape       # B=批次，c=768，h/w=特征图尺寸
        
        # 构建CLIP的影子特征序列：[cls_token + 图像特征展平]
        # 图像特征展平：[B,768,H,W]→[B,768,H×W]→[H×W, B,768]（序列长度为H×W）
        x = torch.cat([cls_token, img_feature.reshape(b, c, -1).permute(2, 0, 1)])
        # x形状：[1 + H×W, B, 768]（1为cls_token，H×W为图像特征序列）

        # 构建sos标记（根据不同格式）
        if self.sos_token_format == 'cls_token':
            # 复用CLIP的cls_token作为sos标记，复制num_sos次
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)  # [100, B, 768]
        elif self.sos_token_format == 'learnable_token':
            # 使用可学习的sos标记
            sos_token = self.sos_token.expand(-1, b, -1)  # [100, B, 768]（扩展到批次维度）
        elif self.sos_token_format == 'pos_embedding':
            # 可学习标记 + cls_token作为sos（结合位置信息）
            sos_token = self.sos_token.expand(-1, b, -1) + cls_token  # [100, B, 768]

        # 构建注意力偏置（适配Transformer层输入格式）
        attn_biases = self._build_attn_biases(bias, target_shape=(h, w))

        # 前向传播：根据是否使用交叉注意力更新sos标记
        if self.cross_attn:
            # 交叉注意力模式：sos_token与图像特征做交叉注意力
            for i, block in enumerate(self.layers):
                sos_token = cross_attn_layer(
                    block,          # Transformer层
                    sos_token,      # 查询（sos标记）
                    x[1:, ],        # 键值（图像特征，不含cls_token）
                    attn_biases[i], # 注意力偏置
                )
                # 除最后一层外，同时更新图像特征
                if i < len(self.layers) - 1:
                    x = block(x)
        else:
            # 自注意力模式：将sos_token与图像特征拼接，做自注意力
            x = torch.cat([sos_token, x], dim=0)  # [100 + 1 + H×W, B, 768]
            for i, block in enumerate(self.layers):
                # 应用Transformer层和注意力偏置
                x = block(x, attn_masks=[attn_biases[i]])
            # 提取更新后的sos_token
            sos_token = x[:self.sos_token_num]  # [100, B, 768]

        # 输出处理：调整维度→归一化→投影→最终归一化
        sos_token = sos_token.permute(1, 0, 2)  # [100, B, 768]→[B, 100, 768]（批次维度前置）
        sos_token = self.ln_post(sos_token)     # 归一化
        sos_token = self.proj(sos_token)        # 投影到out_dims（512）
        if self.final_norm:
            sos_token = F.normalize(sos_token, dim=-1)  # 特征归一化（便于计算余弦相似度）
        
        return sos_token  # [B, 100, 512]：每个掩码提案的特征，用于与文本特征匹配



@MODELS.register_module()  # 注册为模型组件，允许通过配置文件调用
class OPSCLIPHead(BaseDecodeHead):
    """开放全景分割的CLIP头模块，整合SideAdapterNetwork和RecWithAttnbias，实现掩码生成与类别预测
    
    Args:
        num_classes (int): 类别数量（包含背景）
        ops_cfg (ConfigType): SideAdapterNetwork的配置参数
        maskgen_cfg (ConfigType): RecWithAttnbias的配置参数
        deep_supervision_idxs (List[int]): 深度监督的层索引（用于多尺度训练）
        train_cfg (ConfigType): 训练配置（包含匹配和损失计算参数）
    """

    def __init__(self, num_classes: int, ops_cfg: ConfigType,
                 maskgen_cfg: ConfigType, deep_supervision_idxs: List[int],
                 train_cfg: ConfigType, **kwargs):
        # 调用父类BaseDecodeHead的初始化方法
        super().__init__(
            in_channels=ops_cfg.in_channels,  # 输入通道数（与侧网络一致）
            channels=ops_cfg.embed_dims,      # 中间通道数（侧网络嵌入维度）
            num_classes=num_classes,          # 类别数量
           ** kwargs)
        
        # 校验：侧网络的查询数需与识别网络的sos标记数一致（确保掩码提案与类别预测一一对应）
        assert ops_cfg.num_queries == maskgen_cfg.sos_token_num, \
            'num_queries in ops_cfg should be equal to sos_token_num in maskgen_cfg'
        
        # 删除父类默认的分割卷积（OPS用掩码提案+类别分数的方式生成分割结果，无需传统卷积）
        del self.conv_seg
        
        # 初始化侧适配器网络（生成掩码提案和注意力偏置）
        self.side_adapter_network = SideAdapterNetwork(**ops_cfg)
        # 初始化带注意力偏置的识别网络（生成掩码的类别特征）
        self.rec_with_attnbias = RecWithAttnbias(** maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs  # 深度监督的层索引
        self.train_cfg = train_cfg  # 训练配置
        
        # 初始化掩码匹配器（训练时用于将预测掩码与GT匹配）
        if train_cfg:
            self.match_masks = MatchMasks(
                num_points=train_cfg.num_points,  # 点采样数量（用于掩码损失计算）
                num_queries=ops_cfg.num_queries,  # 查询数（掩码提案数量）
                num_classes=num_classes,          # 类别数量
                assigner=train_cfg.assigner)      # 分配器（如匈牙利算法）

    def init_weights(self):
        """初始化权重，支持加载预训练模型的部分参数"""
        rec_state_dict = None  # 识别网络的状态字典
        
        # 若初始化配置为加载部分预训练参数
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') == 'Pretrained_Part':
            # 加载预训练 checkpoint
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            
            # 提取识别网络的参数（过滤掉前缀，只保留rec_with_attnbias的参数）
            rec_state_dict = checkpoint.copy()
            para_prefix = 'decode_head.rec_with_attnbias'  # 预训练参数中的前缀
            prefix_len = len(para_prefix) + 1
            for k, v in checkpoint.items():
                rec_state_dict.pop(k)  # 先删除原有键
                if para_prefix in k:
                    # 保留识别网络的参数（去除前缀）
                    rec_state_dict[k[prefix_len:]] = v

        # 初始化侧网络权重
        self.side_adapter_network.init_weights()
        # 初始化识别网络权重（传入预训练参数）
        self.rec_with_attnbias.init_weights(rec_state_dict)

    def forward(self, inputs: Tuple[Tensor],
                deep_supervision_idxs) -> Tuple[List]:
        """前向传播：生成掩码提案和类别分数
        
        Args:
            inputs (Tuple[Tensor]): 输入三元组
                - imgs: 原始图像
                - clip_feature: CLIP的多尺度视觉特征
                - class_embeds: 文本编码器生成的类别嵌入
            deep_supervision_idxs: 深度监督的层索引
        Returns:
            mask_props (List[Tensor]): 各层的掩码提案，形状为[B, Q, H, W]
            mask_logits (List[Tensor]): 各层的类别分数，形状为[B, Q, C]
        """
        # 解析输入
        imgs, clip_feature, class_embeds = inputs
        
        # 1. 侧网络生成掩码提案和注意力偏置
        mask_props, attn_biases = self.side_adapter_network(
            imgs, clip_feature, deep_supervision_idxs)

        # 2. 识别网络利用注意力偏置生成掩码的类别特征
        mask_embeds = [
            self.rec_with_attnbias(att_bias, clip_feature[-1])
            for att_bias in attn_biases
        ]
        
        # 3. 计算掩码与类别嵌入的相似度（得到类别分数）
        # 通过爱因斯坦求和：[B,Q,C_embed] × [N,C_embed] → [B,Q,N]（N为类别数）
        mask_logits = [
            torch.einsum('bqc,nc->bqn', mask_embed, class_embeds)
            for mask_embed in mask_embeds
        ]
        
        return mask_props, mask_logits

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """预测阶段前向传播：生成最终分割图
        
        Args:
            inputs: 同forward方法
            batch_img_metas: 图像元信息（包含尺寸、缩放因子等）
            test_cfg: 测试配置
        Returns:
            Tensor: 分割概率图，形状为[B, C, H, W]
        """
        # 调用forward获取最后一层的掩码提案和类别分数
        mask_props, mask_logits = self.forward(inputs, [])

        # 用最后一层的结果生成分割图
        return self.predict_by_feat([mask_props[-1], mask_logits[-1]],
                                    batch_img_metas)

    def predict_by_feat(self, seg_logits: List[Tensor],
                        batch_img_metas: List[dict]) -> Tensor:
        """由特征生成分割图：
        1. 将掩码提案上采样到输入图像尺寸
        2. 结合类别分数生成最终分割概率图
        
        Args:
            seg_logits: 包含掩码提案和类别分数的列表
            batch_img_metas: 图像元信息
        Returns:
            Tensor: 分割概率图[B, C, H, W]
        """
        mask_pred = seg_logits[0]  # 掩码提案[B, Q, H, W]
        cls_score = seg_logits[1]  # 类别分数[B, Q, C]
        
        # 获取目标尺寸（填充后的尺寸或原始图像尺寸）
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]  # [H, W]
        else:
            size = batch_img_metas[0]['img_shape'][:2]
        
        # 上采样掩码提案到目标尺寸（双线性插值）
        mask_pred = F.interpolate(
            mask_pred, size=size, mode='bilinear', align_corners=False)
        
        # 类别分数归一化（去除背景类，假设最后一类是背景）
        mask_cls = F.softmax(cls_score, dim=-1)[..., :-1]  # [B, Q, C-1]
        # 掩码提案归一化（sigmoid转为概率）
        mask_pred = mask_pred.sigmoid()  # [B, Q, H, W]
        
        # 生成分割概率图：类别分数 × 掩码提案（按查询维度求和）
        # [B, Q, C-1] × [B, Q, H, W] → [B, C-1, H, W]
        seg_logits = torch.einsum('bqc,bqhw->bchw', mask_cls, mask_pred)
        return seg_logits

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """计算损失：
        1. 将语义分割数据转换为实例数据
        2. 前向传播获取所有深度监督层的输出
        3. 计算并返回损失字典
        
        Args:
            x: 输入数据（图像、CLIP特征、类别嵌入）
            batch_data_samples: 批量数据样本（包含GT语义分割）
            train_cfg: 训练配置
        Returns:
            dict: 损失组件字典
        """
        # 将语义分割数据转换为实例数据（提取每个实例的掩码和类别）
        batch_gt_instances = seg_data_to_instance_data(self.ignore_index,
                                                       batch_data_samples)

        # 前向传播获取所有深度监督层的掩码提案和类别分数
        all_mask_props, all_mask_logits = self.forward(
            x, self.deep_supervision_idxs)

        # 计算损失
        losses = self.loss_by_feat(all_mask_logits, all_mask_props,
                                   batch_gt_instances)

        return losses

    def loss_by_feat(
            self, all_cls_scores: Tensor, all_mask_preds: Tensor,
            batch_gt_instances: List[InstanceData]) -> Dict[str, Tensor]:
        """按特征计算损失：
        1. 对每个深度监督层，将预测掩码与GT实例匹配
        2. 计算类别损失和掩码损失
        3. 整合所有层的损失
        
        Args:
            all_cls_scores: 所有层的类别分数 [num_layers, B, Q, C]
            all_mask_preds: 所有层的掩码提案 [num_layers, B, Q, H, W]
            batch_gt_instances: 批量GT实例数据
        Returns:
            Dict[str, Tensor]: 包含各层损失的字典
        """
        num_dec_layers = len(all_cls_scores)  # 深度监督的层数
        # 为每个层复制一份GT实例（每层独立计算损失）
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]

        losses = []  # 存储每层的损失
        for i in range(num_dec_layers):
            cls_scores = all_cls_scores[i]  # 当前层的类别分数 [B, Q, C]
            mask_preds = all_mask_preds[i]  # 当前层的掩码提案 [B, Q, H, W]
            
            # 匹配：将N个预测掩码与K个GT实例匹配，得到目标标签和掩码
            (labels, mask_targets, mask_weights,
             avg_factor) = self.match_masks.get_targets(
                 cls_scores, mask_preds, batch_gt_instances_list[i])
            
            # 展平类别分数和标签（便于交叉熵计算）
            cls_scores = cls_scores.flatten(0, 1)  # [B×Q, C]
            labels = labels.flatten(0, 1)          # [B×Q]
            
            # 计算平均因子（用于损失归一化）
            num_total_masks = cls_scores.new_tensor([avg_factor],
                                                    dtype=torch.float)
            all_reduce(num_total_masks, op='mean')  # 多卡训练时聚合
            num_total_masks = max(num_total_masks, 1)  # 避免除以0

            # 提取正样本掩码（仅计算匹配成功的掩码损失）
            mask_preds = mask_preds[mask_weights > 0]  # [num_pos, H, W]

            # 若存在正样本，进行点采样（减少计算量，聚焦关键区域）
            if mask_targets.shape[0] != 0:
                with torch.no_grad():
                    # 采样不确定区域的点（结合随机采样和重要性采样）
                    points_coords = get_uncertain_point_coords_with_randomness(
                        mask_preds.unsqueeze(1), None,
                        self.train_cfg.num_points,          # 总采样点数
                        self.train_cfg.oversample_ratio,    # 过采样比例
                        self.train_cfg.importance_sample_ratio)  # 重要性采样比例
                    
                    # 对GT掩码进行点采样 [num_pos, num_points]
                    mask_point_targets = point_sample(
                        mask_targets.unsqueeze(1).float(),
                        points_coords).squeeze(1)
                
                # 对预测掩码进行点采样 [num_pos, num_points]
                mask_point_preds = point_sample(
                    mask_preds.unsqueeze(1), points_coords).squeeze(1)

            # 解析损失函数（支持多个损失组件）
            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            
            layer_loss = dict()  # 当前层的损失
            for loss_decode in losses_decode:
                # 计算类别损失
                if 'loss_cls' in loss_decode.loss_name:
                    if loss_decode.loss_name == 'loss_cls_ce':
                        # 交叉熵损失
                        layer_loss[loss_decode.loss_name] = loss_decode(
                            cls_scores, labels)
                    else:
                        assert False, "分类损失仅支持'CrossEntropyLoss'"

                # 计算掩码损失
                elif 'loss_mask' in loss_decode.loss_name:
                    if mask_targets.shape[0] == 0:
                        # 无正样本时，损失设为0（避免无效计算）
                        layer_loss[loss_decode.loss_name] = mask_preds.sum() * 0.0
                    elif loss_decode.loss_name == 'loss_mask_ce':
                        # 掩码交叉熵损失（基于点采样）
                        layer_loss[loss_decode.loss_name] = loss_decode(
                            mask_point_preds,
                            mask_point_targets,
                            avg_factor=num_total_masks * self.train_cfg.num_points)
                    elif loss_decode.loss_name == 'loss_mask_dice':
                        # 掩码Dice损失（基于点采样）
                        layer_loss[loss_decode.loss_name] = loss_decode(
                            mask_point_preds,
                            mask_point_targets,
                            avg_factor=num_total_masks)
                    else:
                        assert False, "掩码损失仅支持'CrossEntropyLoss'和'DiceLoss'"
                else:
                    assert False, "仅支持'loss_cls'和'loss_mask'类型的损失"

            losses.append(layer_loss)  # 保存当前层损失

        # 整合所有层的损失
        loss_dict = dict()
        # 最后一层的损失直接加入（无前缀）
        loss_dict.update(losses[-1])
        # 其他层的损失添加前缀（如d0.loss_cls_ce）
        for i, loss in enumerate(losses[:-1]):
            for k, v in loss.items():
                loss_dict[f'd{self.deep_supervision_idxs[i]}.{k}'] = v
        
        return loss_dict
    