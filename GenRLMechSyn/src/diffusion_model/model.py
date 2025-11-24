# src/diffusion_model/model.py

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
import math
import logging

# (导入 和 get_beta_schedule 保持不变)
try:
    from .modules import (
        TimestepEmbed,
        PatchEmbed,
        PositionalEncoding,
        BasicTransformerBlock
    )
except ImportError:
    from modules import (
        TimestepEmbed,
        PatchEmbed,
        PositionalEncoding,
        BasicTransformerBlock
    )


def get_beta_schedule(num_timesteps, schedule_type="linear"):
    if schedule_type == "linear":
        return torch.linspace(1e-4, 0.02, num_timesteps)
    elif schedule_type == "cosine":
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0, 0.999)
    else:
        raise NotImplementedError(f"未知的调度表类型: {schedule_type}")


class DiffusionModel(nn.Module):
    """
    (init, _unpatchify, _normalize, _unnormalize, forward,
     _get_tensor_values, q_sample, _x0_from_noise, p_sample ... 均保持不变)
    """

    def __init__(self, config):
        super().__init__()
        self.logger = logging.getLogger()
        model_config = config.get('diffusion_model', {})
        data_config = config.get('data', {})
        self.img_size = model_config.get("img_size", 30)
        self.patch_size = model_config.get("patch_size", 5)
        self.in_channels = model_config.get("in_channels", 5)
        if self.in_channels != 5:
            raise ValueError("模型配置中的 in_channels 必须为 5")
        self.embed_dim = model_config.get("embed_dim", 768)
        self.depth = model_config.get("depth", 12)
        self.num_heads = model_config.get("num_heads", 12)
        self.num_classes = model_config.get("num_classes", 1)
        if self.num_classes <= 0: raise ValueError("num_classes 必须大于 0")
        try:
            norm_values_dict = data_config['normalization_values']
            norm_vec = torch.tensor([
                norm_values_dict['a'],
                norm_values_dict['alpha'],
                norm_values_dict['d']
            ], dtype=torch.float32)
            self.register_buffer('norm_vec', norm_vec.view(1, 3, 1, 1))
            self.norm_value_alpha_max = norm_values_dict['alpha']
        except KeyError:
            raise ValueError("[致命错误] 配置文件中缺少 data.normalization_values 块。")
        if self.img_size % self.patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch 尺寸整除。")
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.time_embed = TimestepEmbed(self.embed_dim)
        self.label_embed = nn.Embedding(self.num_classes, self.embed_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim
        )
        self.pos_embed = PositionalEncoding(
            num_patches=self.num_patches,
            embed_dim=self.embed_dim
        )
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
            for _ in range(self.depth)
        ])
        self.final_norm = nn.LayerNorm(self.embed_dim)
        patch_dim = (self.patch_size ** 2) * self.in_channels
        self.final_linear = nn.Linear(self.embed_dim, patch_dim)
        self.num_timesteps = model_config.get("timesteps", 1000)
        betas = get_beta_schedule(self.num_timesteps, model_config.get("schedule_type", "cosine"))
        betas = betas.to(torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev_init = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=torch.float32), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev_init)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        posterior_variance_init = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', torch.clamp(posterior_variance_init, min=1e-20))
        posterior_mean_coef1_init = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1_init)
        posterior_mean_coef2_init = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2_init)
        self.logger.info(f"DiffusionTransformer (DiT) 模型已初始化 (输入通道: {self.in_channels}):")
        self.logger.info(f"  Embed Dim: {self.embed_dim}, Depth: {self.depth}, Heads: {self.num_heads}")
        self.logger.info(
            f"  Image: {self.img_size}x{self.img_size}, Patch: {self.patch_size}, Patches: {self.num_patches}")
        self.logger.info(f"  Normalization values (a, alpha, offset): {self.norm_vec.squeeze().tolist()}")

    def _unpatchify(self, x):
        P = self.patch_size
        C = self.in_channels
        H = W = self.img_size
        num_patches_h = H // P
        num_patches_w = W // P
        x = rearrange(
            x, 'b (nh nw) (p ph c) -> b c (nh p) (nw ph)',
            c=C, p=P, ph=P, nh=num_patches_h, nw=num_patches_w
        )
        return x

    def _normalize(self, x_tensor):
        if x_tensor.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor.shape[1]}")
        exists_channel = x_tensor[:, 0:1, :, :]
        joint_type_channel = x_tensor[:, 1:2, :, :]
        other_channels = x_tensor[:, 2:, :, :]
        exists_norm = exists_channel * 2.0 - 1.0
        joint_type_norm = joint_type_channel
        other_norm = (other_channels / (self.norm_vec + 1e-8)) * 2.0 - 1.0
        return torch.cat([exists_norm, joint_type_norm, other_norm], dim=1)

    def _unnormalize(self, x_tensor_norm):
        if x_tensor_norm.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor_norm.shape[1]}")
        exists_norm = x_tensor_norm[:, 0:1, :, :]
        joint_type_norm = x_tensor_norm[:, 1:2, :, :]
        other_norm = x_tensor_norm[:, 2:, :, :]
        exists_unnorm = (exists_norm + 1.0) / 2.0
        joint_type_unnorm = joint_type_norm
        other_unnorm = ((other_norm + 1.0) / 2.0) * self.norm_vec
        return torch.cat([exists_unnorm, joint_type_unnorm, other_unnorm], dim=1)

    def forward(self, x_norm, t, y):
        t_embed = self.time_embed(t)
        y_embed = self.label_embed(y)
        c = t_embed + y_embed
        x = self.patch_embed(x_norm)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)
        x = self.final_linear(x)
        noise_pred_norm = self._unpatchify(x)
        return noise_pred_norm

    def _get_tensor_values(self, val, t):
        batch_size = t.shape[0]
        t_long = t.long().to(val.device)
        out = val.gather(-1, t_long)
        return out.reshape(batch_size, 1, 1, 1)

    def q_sample(self, x_start_norm, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start_norm)
        sqrt_alphas_cumprod_t = self._get_tensor_values(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)
        x_t_norm = sqrt_alphas_cumprod_t * x_start_norm + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t_norm

    def _x0_from_noise(self, x_t_norm, t, noise_norm):
        sqrt_recip_alphas_cumprod_t = self._get_tensor_values(self.alphas_cumprod.sqrt().reciprocal(), t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)
        x_0_pred_norm = sqrt_recip_alphas_cumprod_t * (x_t_norm - sqrt_one_minus_alphas_cumprod_t * noise_norm)
        return x_0_pred_norm

    def p_sample(self, x_t_norm, t, t_index, y, guidance_fn=None, guidance_scale=1.0):
        betas_t = self._get_tensor_values(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = self._get_tensor_values(self.sqrt_recip_alphas, t)
        predicted_noise_norm = self.forward(x_t_norm, t, y)
        if guidance_fn is not None:
            gradient = guidance_fn(x_t_norm, t, y)
            predicted_noise_norm = predicted_noise_norm - sqrt_one_minus_alphas_cumprod_t * guidance_scale * gradient
        x_0_pred_norm = self._x0_from_noise(x_t_norm, t, predicted_noise_norm)
        dynamic_threshold_percentile = 0.995
        abs_x0_flat = x_0_pred_norm.abs().view(-1)
        s = torch.quantile(abs_x0_flat, dynamic_threshold_percentile)
        s = torch.max(s, torch.tensor(1.0, device=s.device))
        x_0_pred_norm = torch.clamp(x_0_pred_norm, -s, s)
        posterior_mean_coef1_t = self._get_tensor_values(self.posterior_mean_coef1, t)
        posterior_mean_coef2_t = self._get_tensor_values(self.posterior_mean_coef2, t)
        posterior_mean = posterior_mean_coef1_t * x_0_pred_norm + posterior_mean_coef2_t * x_t_norm
        posterior_variance_t = self._get_tensor_values(self.posterior_variance, t)
        noise = torch.randn_like(x_t_norm)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t_norm.shape) - 1)))
        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    # --- 物理约束应用 (包含钳位) ---
    def apply_physics_constraints(self, x_norm, clamp=True):
        """
        将归一化的输出转换为符合物理约束的张量。
        [修改说明] 对 alpha 使用取余操作以适应角度周期性。
        """
        # 1. 反归一化
        x_phys = self._unnormalize(x_norm)
        exists, j_type, a, alpha, offset = torch.chunk(x_phys, 5, dim=1)

        # 2. 强制对称性 (M + M.T) / 2
        exists = (exists + exists.transpose(2, 3)) / 2.0
        a = (a + a.transpose(2, 3)) / 2.0
        alpha = (alpha + alpha.transpose(2, 3)) / 2.0

        # 3. 关节类型行一致性
        j_type_mean = j_type.mean(dim=3, keepdim=True)
        j_type = j_type_mean.expand_as(j_type)

        # 4. 强制非连接处为 0 (Masking)
        link_mask = (exists > 0.5).float()
        a = a * link_mask
        alpha = alpha * link_mask
        offset = offset * link_mask

        # 5. [可选] 物理参数钳位/规范化
        if clamp:
            # 连杆长度 a 必须非负，使用 clamp
            a = torch.abs(a)

            # [修改] 扭转角 alpha 使用取余，保证在 [0, 2*pi)
            # 这样 -10度 会自动变为 350度，370度 变为 10度
            alpha = alpha % (2 * math.pi)

            # offset (d) 可以为负，不钳位

        # 6. 强制对角线清零
        B, _, H, W = exists.shape
        device = x_norm.device
        diag_mask = 1.0 - torch.eye(H, device=device).view(1, 1, H, W)

        exists = exists * diag_mask
        a = a * diag_mask
        alpha = alpha * diag_mask
        offset = offset * diag_mask

        # 7. 重新组合
        return torch.cat([exists, j_type, a, alpha, offset], dim=1)

    # --- 采样函数 (sample) ---
    @torch.no_grad()
    def sample(self, num_samples, y, guidance_fn=None, guidance_scale=1.0):
        self.logger.info(f"开始采样 {num_samples} 个新机构...")
        device = next(self.parameters()).device

        shape = (num_samples, self.in_channels, self.img_size, self.img_size)
        x_t_norm = torch.randn(shape, device=device)

        for t_index in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM 采样", total=self.num_timesteps):
            t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
            x_t_norm = self.p_sample(x_t_norm, t, t_index, y, guidance_fn, guidance_scale)

        x_0_norm = x_t_norm

        # 使用统一的物理约束处理 (Masking + Clamping)
        x_phys_clean = self.apply_physics_constraints(x_0_norm)

        # 最终离散化处理 (仅用于输出 Numpy)
        e, j, a, al, o = torch.chunk(x_phys_clean, 5, dim=1)
        e = (e > 0.5).float()
        j = torch.where(j > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

        # 最终输出组合
        x_final = torch.cat([e, j, a, al, o], dim=1)

        x_0_numpy = x_final.permute(0, 2, 3, 1).cpu().numpy()
        generated_mechanisms = [x_0_numpy[i] for i in range(num_samples)]

        return generated_mechanisms, x_0_norm
