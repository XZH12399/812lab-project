# src/diffusion_model/modules.py

import torch
import torch.nn as nn
import math
from einops import rearrange


# -----------------------------------------------------------------
# 模块 1: 时间步嵌入 (Timestep Embedding)
# 作用: 将整数时间步 t 转换为一个高维向量 c
# -----------------------------------------------------------------

class TimestepEmbed(nn.Module):
    """
    将整数时间步 t 编码为 (B, D) 的向量。
    这是来自 "Attention Is All You Need" 的标准正弦位置编码。
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 定义一个MLP，将正弦编码投影到最终的嵌入维度
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),  # SiLU (Swish) 激活函数
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t):
        """
        输入: t (B,) - 批量的整数时间步
        输出: c (B, D) - 嵌入后的时间向量
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.embed_dim // 2
        # 计算 10000^(-2i / D)
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        freqs = freqs.to(device=t.device)

        # 计算 t * freqs
        args = t[:, None].float() * freqs[None, :]

        # 计算 sin 和 cos
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 如果维度 D 是奇数, 补一个0
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # 通过MLP投影
        return self.mlp(embedding)


# -----------------------------------------------------------------
# 模块 2: Patch 嵌入 (PatchEmbed)
# 作用: 将 (B, 3, 30, 30) 的 "图像" 转换为 (B, N, D) 的 "Token序列"
# -----------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    将 (B, C, H, W) 的图像分割成 (B, N, D) 的 patch 序列。
    N = H*W / (P*P) 是 patch 的数量。
    """

    def __init__(self, img_size=30, patch_size=5, in_channels=3, embed_dim=768):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch 尺寸整除。")

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        # 关键: 使用一个卷积层来实现 "Patchify"
        # kernel_size 和 stride 都等于 patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        输入: x (B, C, H, W) - e.g., (B, 3, 30, 30)
        输出: x (B, N, D) - e.g., (B, 36, 768)
        """
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"输入图像尺寸 ({H}, {W}) 与模型期望的 ({self.img_size}) 不符。")

        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)

        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        # einops.rearrange 是一个强大的重排工具
        # 'b d h w -> b (h w) d' 的意思是:
        # b=batch, d=embed_dim, h=height, w=width
        # 保持 b 和 d 不变, 将 h 和 w 合并为 N = (h w)
        # 然后将 (b d N) 变为 (b N d)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


# -----------------------------------------------------------------
# 模块 3: 位置编码 (PositionalEncoding)
# 作用: 告诉 Transformer 每一个 patch 在原始图像中的 "位置"
# -----------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    可学习的2D位置编码。
    (B, N, D)
    """

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # 创建一个 (1, N, D) 的可学习参数
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * .02)

    def forward(self, x):
        """
        输入: x (B, N, D)
        输出: x (B, N, D) (加上了位置编码)
        """
        # x + pos_embed
        # (B, N, D) + (1, N, D) -> 广播机制
        return x + self.pos_embed


# -----------------------------------------------------------------
# 模块 4: DiT 核心 - adaLN-Zero (自适应层归一化)
# 作用: 将时间向量 c "注入" 到 Transformer 块中
# -----------------------------------------------------------------

class adaLN_Zero(nn.Module):
    """
    DiT 论文中提出的自适应层归一化 (adaLN-Zero) 模块。
    它接收时间嵌入 c, 并预测 LayerNorm 的 scale 和 shift,
    同时还预测一个控制块输出的门控参数 alpha.
    """

    def __init__(self, embed_dim):
        super().__init__()
        # DiT 的核心: LayerNorm
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        # 一个线性层, 用于从时间嵌入 c 中预测 scale, shift 和 alpha
        # (D) -> (6*D)
        # 我们需要 2*D 用于 LayerNorm (scale, shift)
        # 我们需要 2*D 用于 Attention 块
        # 我们需要 2*D 用于 MLP 块
        self.linear = nn.Linear(embed_dim, embed_dim * 6)

        # 初始化为0, 确保在训练初期,
        # 这个块是一个恒等映射 (Identity map)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        """
        输入:
        x (B, N, D) - 来自前一层的Token序列
        c (B, D)    - 时间嵌入向量

        输出:
        normed_x (B, N, D) - 归一化后的x
        (scale_att, shift_att, gate_att,
         scale_mlp, shift_mlp, gate_mlp) - 6个控制参数
        """
        # c (B, D) -> (B, 6*D)
        c = self.linear(c)

        # (B, 6*D) -> 6 * (B, D)
        # 我们使用 torch.chunk 将其分割
        # 每个都是 (B, 1, D) 的形状, 方便广播
        scale_att, shift_att, gate_att, \
        scale_mlp, shift_mlp, gate_mlp = (c[:, None, :]).chunk(6, dim=2)

        # 对 x 进行 LayerNorm
        # (B, N, D) -> (B, N, D)
        # (1 + scale) * norm(x) + shift
        normed_x = self.norm(x) * (1 + scale_att) + shift_att

        return normed_x, gate_att, scale_mlp, shift_mlp, gate_mlp


# -----------------------------------------------------------------
# 模块 5: 基础 Transformer 块
# -----------------------------------------------------------------

class BasicTransformerBlock(nn.Module):
    """
    一个完整的 DiT 块, 包含 自注意力 和 MLP,
    并使用 adaLN_Zero 进行条件注入。
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 自适应归一化 (注意力之前)
        self.adaLN_1 = adaLN_Zero(embed_dim)

        # 多头自注意力 (MHA)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True  # 确保输入是 (B, N, D)
        )

        # 自适应归一化 (MLP之前)
        self.adaLN_2 = adaLN_Zero(embed_dim)

        # MLP (前馈网络)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, c):
        """
        输入:
        x (B, N, D) - Token序列
        c (B, D)    - 时间嵌入

        输出:
        x (B, N, D) - 处理后的Token序列
        """
        # 1. 自适应归一化
        normed_x_att, gate_att, scale_mlp, shift_mlp, gate_mlp = self.adaLN_1(x, c)

        # 2. 自注意力
        # MHA 返回 (attn_output, attn_weights)
        attn_output, _ = self.attention(normed_x_att, normed_x_att, normed_x_att)

        # 3. 残差连接 (使用门控)
        # gate_att 使得模型可以学习跳过这个块
        x = x + gate_att * attn_output

        # 4. 第二次自适应归一化
        normed_x_mlp = self.adaLN_2.norm(x) * (1 + scale_mlp) + shift_mlp

        # 5. MLP
        mlp_output = self.mlp(normed_x_mlp)

        # 6. 第二次残差连接 (使用门控)
        x = x + gate_mlp * mlp_output

        return x