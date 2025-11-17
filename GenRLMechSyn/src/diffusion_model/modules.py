# src/diffusion_model/modules.py

import torch
import torch.nn as nn
import math
from einops import rearrange

# ... (TimestepEmbed 保持不变) ...
class TimestepEmbed(nn.Module):
    """
    将整数时间步 t 编码为 (B, D) 的向量。
    这是来自 "Attention Is All You Need" 的标准正弦位置编码。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),  # SiLU (Swish) 激活函数
            nn.Linear(embed_dim * 4, embed_dim),
        )
    def forward(self, t):
        if t.dim() == 0: t = t.unsqueeze(0)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)
# -----------------------------------------------------------------
# 模块 2: Patch 嵌入 (PatchEmbed)
# -----------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    将 (B, C, H, W) 的图像分割成 (B, N, D) 的 patch 序列。
    N = H*W / (P*P) 是 patch 的数量。
    """

    def __init__(self, img_size=30, patch_size=5, in_channels=4, embed_dim=768): # <-- 默认 4
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch 尺寸整除。")

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        # 关键: 使用一个卷积层来实现 "Patchify"
        self.proj = nn.Conv2d(
            in_channels, # <-- 使用 in_channels (4)
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        输入: x (B, C, H, W) - e.g., (B, 4, 30, 30)
        输出: x (B, N, D) - e.g., (B, 36, 768)
        """
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"输入图像尺寸 ({H}, {W}) 与模型期望的 ({self.img_size}) 不符。")
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

# ... (PositionalEncoding, adaLN_Zero, BasicTransformerBlock 保持不变) ...
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * .02)
    def forward(self, x):
        return x + self.pos_embed

class adaLN_Zero(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.linear = nn.Linear(embed_dim, embed_dim * 6)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    def forward(self, x, c):
        c = self.linear(c)
        scale_att, shift_att, gate_att, \
        scale_mlp, shift_mlp, gate_mlp = (c[:, None, :]).chunk(6, dim=2)
        normed_x = self.norm(x) * (1 + scale_att) + shift_att
        return normed_x, gate_att, scale_mlp, shift_mlp, gate_mlp

class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.adaLN_1 = adaLN_Zero(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.adaLN_2 = adaLN_Zero(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    def forward(self, x, c):
        normed_x_att, gate_att, scale_mlp, shift_mlp, gate_mlp = self.adaLN_1(x, c)
        attn_output, _ = self.attention(normed_x_att, normed_x_att, normed_x_att)
        x = x + gate_att * attn_output
        normed_x_mlp = self.adaLN_2.norm(x) * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp * mlp_output
        return x