# src/diffusion_model/model.py

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
import math
import logging

# 导入我们创建的 "零件"
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


# -----------------------------------------------------------------
# 辅助函数: 噪声调度表 (Beta Schedule)
# -----------------------------------------------------------------
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


# -----------------------------------------------------------------
# 核心模型: DiffusionModel (DiT)
# -----------------------------------------------------------------

class DiffusionModel(nn.Module):
    """
    DiT (Diffusion Transformer) 模型.
    使用 5 通道输入: [exists, joint_type, a, alpha, offset].
    """

    def __init__(self, config):
        """
        初始化模型.
        config 是一个字典, 来源于 'configs/default_config.yaml'
        """
        super().__init__()

        # --- 获取 Logger ---
        self.logger = logging.getLogger()

        # --- 读取配置 ---
        model_config = config.get('diffusion_model', {})
        data_config = config.get('data', {})

        self.img_size = model_config.get("img_size", 30)
        self.patch_size = model_config.get("patch_size", 5)

        self.in_channels = model_config.get("in_channels", 5)  # 默认 5
        if self.in_channels != 5:
            raise ValueError("模型配置中的 in_channels 必须为 5 ([exists, joint_type, a, alpha, offset])")

        self.embed_dim = model_config.get("embed_dim", 768)
        self.depth = model_config.get("depth", 12)
        self.num_heads = model_config.get("num_heads", 12)

        # --- 读取类别数量 ---
        self.num_classes = model_config.get("num_classes", 1)
        if self.num_classes <= 0: raise ValueError("num_classes 必须大于 0")

        # --- 加载按通道的归一化值 ---
        try:
            norm_values_dict = data_config['normalization_values']
            # 我们需要 [a, alpha, d] (d 即 offset) 三个通道的值
            norm_vec = torch.tensor([
                norm_values_dict['a'],
                norm_values_dict['alpha'],
                norm_values_dict['d']  # config 中的 key 保持 'd'
            ], dtype=torch.float32)  # shape [3]

            # 注册为 buffer (1, 3, 1, 1), 以便在 _normalize 中广播
            self.register_buffer('norm_vec', norm_vec.view(1, 3, 1, 1))
            self.norm_value_alpha_max = norm_values_dict['alpha']

        except KeyError:
            raise ValueError("[致命错误] 配置文件中缺少 data.normalization_values 块。")

        # --- 组装 DiT 架构 ---
        if self.img_size % self.patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch 尺寸整除。")
        self.num_patches = (self.img_size // self.patch_size) ** 2

        self.time_embed = TimestepEmbed(self.embed_dim)
        self.label_embed = nn.Embedding(self.num_classes, self.embed_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,  # <-- 使用 5
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

        # --- 解码器线性层: (D) -> (P*P*C=5) ---
        patch_dim = (self.patch_size ** 2) * self.in_channels  # <-- C=5
        self.final_linear = nn.Linear(self.embed_dim, patch_dim)

        # --- 设置扩散过程参数 ---
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

    # --- Unpatchify 辅助函数 ---
    def _unpatchify(self, x):
        """将 Token 序列 (B, N, P*P*C=5) 转换回图像 (B, C=5, H, W)."""
        P = self.patch_size
        C = self.in_channels  # <-- C=5
        H = W = self.img_size
        num_patches_h = H // P
        num_patches_w = W // P

        x = rearrange(
            x, 'b (nh nw) (p ph c) -> b c (nh p) (nw ph)',
            c=C, p=P, ph=P, nh=num_patches_h, nw=num_patches_w
        )
        return x

    # --- 归一化/反归一化 (处理5通道) ---
    def _normalize(self, x_tensor):
        """ 将 (B, 5, H, W) 数据按通道归一化到 [-1, 1] 范围."""
        if x_tensor.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor.shape[1]}")

        exists_channel = x_tensor[:, 0:1, :, :]  # (B, 1, H, W) (0/1)
        joint_type_channel = x_tensor[:, 1:2, :, :]  # (B, 1, H, W) (-1/1)
        other_channels = x_tensor[:, 2:, :, :]  # (B, 3, H, W) (a, alpha, offset)

        exists_norm = exists_channel * 2.0 - 1.0  # 0/1 -> -1/1

        # joint_type 已经是 -1/1, 无需操作
        joint_type_norm = joint_type_channel

        # 按通道归一化 [a, alpha, offset]: (B, 3, H, W) / (1, 3, 1, 1)
        other_norm = (other_channels / (self.norm_vec + 1e-8)) * 2.0 - 1.0

        return torch.cat([exists_norm, joint_type_norm, other_norm], dim=1)

    def _unnormalize(self, x_tensor_norm):
        """ 将 (B, 5, H, W) 数据从 [-1, 1] 按通道恢复到原始范围."""
        if x_tensor_norm.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor_norm.shape[1]}")

        exists_norm = x_tensor_norm[:, 0:1, :, :]  # (B, 1, H, W)
        joint_type_norm = x_tensor_norm[:, 1:2, :, :]  # (B, 1, H, W)
        other_norm = x_tensor_norm[:, 2:, :, :]  # (B, 3, H, W)

        exists_unnorm = (exists_norm + 1.0) / 2.0  # -1/1 -> 0/1

        # joint_type 保持 -1/1
        joint_type_unnorm = joint_type_norm

        # 按通道反归一化 [a, alpha, offset]: (B, 3, H, W) * (1, 3, 1, 1)
        other_unnorm = ((other_norm + 1.0) / 2.0) * self.norm_vec

        return torch.cat([exists_unnorm, joint_type_unnorm, other_unnorm], dim=1)

    # --- 前向传播 (接收标签 y) ---
    def forward(self, x_norm, t, y):
        """
        接收 *归一化* 的 x_t_norm (B, 5, H, W), t (B,), 和标签 y (B,),
        预测 *归一化* 的噪声.
        """
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

    # --- 扩散过程辅助函数 ---
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

    # --- 反向去噪 (p_sample, 无中间钳位) ---
    def p_sample(self, x_t_norm, t, t_index, y, guidance_fn=None, guidance_scale=1.0):
        # 在归一化空间操作, 通道数不影响它
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

    # --- 采样函数 (sample, 在末尾钳位) ---
    @torch.no_grad()
    def sample(self, num_samples, y, guidance_fn=None, guidance_scale=1.0):
        """
        完整的采样循环. **接收标签 y**.
        返回: (generated_mechanisms_list, pure_x0_norm_batch)
        """
        self.logger.info(f"开始采样 {num_samples} 个新机构 (格式: [exists, joint_type, a, alpha, offset])...")
        device = next(self.parameters()).device

        # 1. 从纯噪声 x_T 开始 (归一化空间)
        shape = (num_samples, self.in_channels, self.img_size, self.img_size)  # C=5
        x_t_norm = torch.randn(shape, device=device)

        # 2. 从 T 循环到 1
        for t_index in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM 采样", total=self.num_timesteps):
            t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
            x_t_norm = self.p_sample(x_t_norm, t, t_index, y, guidance_fn, guidance_scale)

        # 3. x_0_norm 就是最终的 x_t_norm (B, 5, H, W)
        #    这是我们需要的 "纯粹" x_0_norm (在GPU上)
        x_0_norm = x_t_norm

        # 4. 在采样结束后进行【一次性】反归一化和钳位
        # 4a. 反归一化
        x_0_unnorm = self._unnormalize(x_0_norm)  # (B, 5, H, W)

        # 4b. 分离通道
        exists_unnorm, joint_type_unnorm, a_unnorm, alpha_unnorm, offset_unnorm = torch.chunk(x_0_unnorm, 5, dim=1)

        # 4c. 钳位 exists 通道 (阈值 0.5)
        exists_clamped = (exists_unnorm > 0.5).float()  # 变为 0.0 或 1.0

        # 4c-bis. 钳位 joint_type 通道 (阈值 0)
        # > 0 变为 +1 (R), <= 0 变为 -1 (P)
        joint_type_clamped = torch.where(joint_type_unnorm > 0, 1.0, -1.0)

        # 4d. 钳位 a, alpha, offset 通道 (物理约束)
        a_clamped = torch.clamp(a_unnorm, min=0.0)
        alpha_clamped = torch.clamp(alpha_unnorm, min=0.0, max=math.pi / 2.0)
        offset_clamped = torch.clamp(offset_unnorm, min=0.0)  # (B, 1, H, W)

        # --- 步骤 4d-bis: 应用轴向平移不变性 ---
        exists_mask = (exists_clamped > 0.5)

        offset_for_min_finding = offset_clamped.clone()
        offset_for_min_finding[~exists_mask] = float('inf')

        min_offsets_per_axis = torch.min(offset_for_min_finding, dim=3, keepdim=True)[0]

        min_offsets_per_axis = torch.where(
            torch.isinf(min_offsets_per_axis),
            0.0,
            min_offsets_per_axis
        )

        offset_clamped = offset_clamped - min_offsets_per_axis

        # 4e. 如果 exists=0, 强制 a,alpha,offset,joint_type 为 0
        a_clamped = a_clamped * exists_clamped
        alpha_clamped = alpha_clamped * exists_clamped
        offset_clamped = offset_clamped * exists_clamped
        joint_type_clamped = joint_type_clamped * exists_clamped

        # 4f. 重新组合通道 (最终的、钳位后的、未归一化的结果)
        x_0_final_unnorm = torch.cat([
            exists_clamped,
            joint_type_clamped,
            a_clamped,
            alpha_clamped,
            offset_clamped
        ], dim=1)

        # 5. 转换 (B, C, H, W) -> (B, H, W, C) 并移到 CPU
        x_0_numpy = x_0_final_unnorm.permute(0, 2, 3, 1).cpu().numpy()

        # 6. 分割成列表
        generated_mechanisms = [x_0_numpy[i] for i in range(num_samples)]

        self.logger.info(f"采样完成. 共生成 {len(generated_mechanisms)} 个机构张量 (已应用最终钳位)。")

        # --- 返回两个值 ---
        # 1. 列表: 后处理的 numpy 数组 (用于评估和保存)
        # 2. 批张量: 纯粹的 x_0_norm (用于 RL 缓冲区)
        return generated_mechanisms, x_0_norm