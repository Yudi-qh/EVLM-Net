# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import sys
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
sam2_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
if sam2_path not in sys.path:
    sys.path.append(sam2_path)

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

# class AdapterInBlock(nn.Module):
#     def __init__(self,dim,adapter_type):
#         super(AdapterInBlock, self).__init__()
#         # self.block=blk
#         self.dim=dim
#         self.adapter_type=adapter_type
#         # if self.adapter_type=="adapter_attn":
#         #     dim=self.block.shape[-1]
#         # if self.adapter_type=="adapter_mlp":
#         #     dim=self.block.shape[-1]
#
#
#         self.prompt_learn = nn.Sequential(
#             nn.Linear(dim, 32),
#             nn.GELU(),
#             nn.Linear(32, dim),
#             nn.GELU()
#         )
#     def forward(self, x):
#         prompt=self.prompt_learn(x)
#         promped=x+prompt
#         # net=self.block(promped)
#         return prom


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )


        # 定义Adapter层
        # self.adapter_attn=nn.Sequential(
        #     nn.Linear(dim_out, 32),
        #     nn.GELU(),
        #     nn.Linear(32, dim_out),
        #     nn.GELU()
        # )
        #
        # self.adapter_mlp = nn.Sequential(
        #     nn.Linear(dim_out, 32),
        #     nn.GELU(),
        #     # nn.ReLU(),
        #     nn.Linear(32, dim_out),
        #     nn.GELU()
        #     # nn.ReLU(),
        # )

        # nn.init.xavier_uniform_(self.adapter_attn.weight)
        # nn.init.zeros_(self.adapter_attn.bias)
        # nn.init.xavier_uniform_(self.adapter_mlp.weight)
        # nn.init.zeros_(self.adapter_mlp.bias)

        # for layer in self.adapter_attn:
        #     if isinstance(layer, nn.Linear):
        #         # 对 Linear 层使用 Xavier 初始化
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)  # 对 bias 使用零初始化
        #
        # for layer in self.adapter_mlp:
        #     if isinstance(layer, nn.Linear):
        #         # 对 Linear 层使用 Xavier 初始化
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.zeros_(layer.bias)  # 对 bias 使用零初始化

        # self.adapter_attn = AdapterInBlock(dim_out, "adapter_attn")
        # self.adapter_mlp = AdapterInBlock(dim_out, "adapter_mlp")

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        # print("\n")
        # print(f"shortcut:{shortcut.shape}")
        x = self.norm1(x)
        # print(f"norm1:{x.shape}")

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
            # print(f"输入输出维度不同时shortcut：{shortcut.shape}")

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)

        # x=self.attn(x)

        x = self.attn(x)
        # print(f"x_attn:{x.shape}")

        # attn后添加adapter层
        # x=x+self.adapter_attn(x)

        # x=x+self.adapter_attn(x)

        # print(f"残差连接后的adapter_attn:{x.shape}")

        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            # x=window_unpartition(x,window_size,pad_hw,(H,W))
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        # 残差连接
        # x=shortcut+self.drop_path(x)
        # print(f"attn和drop_path之后的输出：{x.shape}")
        # print(f"第一个残差连接部分：shortcut：{shortcut.shape},x：{x_attn.shape}")
        x = shortcut + self.drop_path(x)
        shortcut2=x

        # MLP
        x=self.mlp(self.norm2(x))
        # print(f"通过mlp层后：{x_mlp.shape}")


        # x=x+self.adapter_mlp(x)


        # 残差连接
        # print(f"第二个残差连接部分：shortcut:{x.shape},x_mlp:{x_mlp.shape}")
        x=shortcut2+self.drop_path(x)
        # print(f"encoder block的输出：{x.shape}")

        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print(f"encoder block的最后输出：{x.shape}")
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,

        # biomedclip_hf_api: str,  # 从hugging face加载biomedclip的路径

        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )


        # 加载biomedclip模型
        # self.biomedclip=open_clip.create_model(biomedclip_hf_api)


        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed




    #   处理文本输入
    # def _forward_bert(self,
    #                   x,
    #                   # 注意力掩码如果没提供会自动计算
    #                   attention_mask: Optional[torch.LongTensor] = None,
    #                   # 是否返回所有隐层的状态，false代表只返回最后一层的输出
    #                   output_hidden_states: bool = False,
    #                   ):
    #     # 从 biomedclip 模型中获取文本编码器部分
    #     bert=self.biomedclip.text
    #     # 检查是否传入了 attention_mask，如果没有传入，代码会自动生成。
    #     # 生成方式是：将输入 x 中的每个 token 与 pad_token_id（BERT 配置中的填充 token 的 ID）进行比较，得到一个布尔值的 mask，表明哪些位置是填充的。
    #     if attention_mask is None:
    #         attention_mask = (x != bert.config.pad_token_id).long()
    #
    #     # 将输入的文本 x 和 attention_mask 传递给 BERT 模型的 transformer 部分。
    #     out = bert.transformer(
    #         input_ids=x,
    #         attention_mask=attention_mask,
    #         output_hidden_states=output_hidden_states,
    #     )
    #
    #     # BERT 通常有一个池化操作（pooler），它将 transformer 的输出（通常是最后一层的输出）进行池化，通常是提取 [CLS] 标记对应的向量并进行一些处理，作为整个句子的表示。
    #     pooled_out = bert.pooler(out, attention_mask)
    #
    #     # proj 是一个投影层，用于将池化后的输出映射到一个新的空间。这是为了确保输出维度与模型配置中定义的维度一致
    #     projected = bert.proj(pooled_out)
    #
    #     # 获取 transformer 输出的最后一层隐藏状态的形状，这里 seq_len 是文本序列的长度
    #     seq_len = out.last_hidden_state.shape[1]
    #     tokens = (
    #         out.last_hidden_state[
    #         :, torch.arange(seq_len) != bert.pooler.cls_token_position, :
    #         ]
    #         if type(bert.pooler) == ClsPooler
    #         else out.last_hidden_state
    #     )
    #
    #     if bert.output_tokens:
    #         return projected, tokens
    #
    #     if output_hidden_states:
    #         return projected, out.hidden_states
    #     else:
    #         return projected

    # def get_conditional_embeddings(
    #         self,
    #         batch_size: int,
    #         # input_ids: 输入文本的 ID，形状为 (batch_size, seq_len)
    #         input_ids: torch.Tensor,
    #         attention_mask: torch.Tensor, #值为 1 表示该位置有效，0 表示该位置是填充的。
    # ):
    #
    #
    #     # compute conditional embeddings from texts
    #     # 检查输入的 input_ids 的第一个维度（即文本的数量）是否与 batch_size 一致
    #     if len(input_ids) != batch_size:
    #         raise ValueError(
    #             "Make sure to pass as many prompt texts as there are query images"
    #         )
    #     conditional_embeddings = self._forward_bert(
    #         input_ids, attention_mask=attention_mask, output_hidden_states=False
    #     )
    #
    #     # 返回经过 BERT 编码后的文本特征
    #     return conditional_embeddings


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        # conditional_embeddings = self.get_conditional_embeddings(
        #     batch_size=x.shape[0],
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        # )


        # print(f"一开始的x：{x.shape}")
        x = self.patch_embed(x)
        # print(f"patch_embed后：{x.shape}")
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # print(f"加上pos embed后：{x.shape}")

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
