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
    # print("警告: 无法从 .modules 导入, 尝试从 modules 导入 (适用于本地测试)")
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
    """
    创建扩散过程的噪声调度表 (betas).
    """
    if schedule_type == "linear":
        # 线性调度: 从 1e-4 上升到 0.02
        return torch.linspace(1e-4, 0.02, num_timesteps)
    elif schedule_type == "cosine":
        # Cosine 调度 (效果通常更好)
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
    一个完整的 DiT (Diffusion Transformer) 模型.
    它包含了神经网络架构和扩散/采样逻辑.
    使用 4 通道输入: [exists, a, alpha, d].
    (钳位操作移至 sample 函数末尾)
    (已更新!) DiT 模型, 支持类别条件 (Class Conditioning).
    条件 c = t_embed + y_embed.
    """
    def __init__(self, config):
        """
        初始化模型.
        config 是一个字典, 来源于 'configs/default_config.yaml'
        """
        super().__init__()

        # --- 获取 Logger ---
        self.logger = logging.getLogger()  # 获取根 logger (已在 train.py 中配置)

        # --- 读取配置 ---
        model_config = config.get('diffusion_model', {})  # 添加默认空字典防止 KeyErrors
        data_config = config.get('data', {})

        self.img_size = model_config.get("img_size", 30)
        self.patch_size = model_config.get("patch_size", 5)
        self.in_channels = model_config.get("in_channels", 4)
        if self.in_channels != 4:
            raise ValueError("模型配置中的 in_channels 必须为 4 ([exists, a, alpha, d])")

        self.embed_dim = model_config.get("embed_dim", 768)
        self.depth = model_config.get("depth", 12)
        self.num_heads = model_config.get("num_heads", 12)

        # --- 读取类别数量 ---
        self.num_classes = model_config.get("num_classes", 1)
        if self.num_classes <= 0: raise ValueError("num_classes 必须大于 0")

        # --- 存储归一化值 ---
        try:
            # 确保从 data_config 读取
            self.norm_value = data_config['normalization_value']
        except KeyError:
            self.logger.warning("[警告] 配置文件中缺少 data.normalization_value, 将使用默认值 10.0")
            self.norm_value = 10.0

        # --- 组装 DiT 架构 ---
        if self.img_size % self.patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch 尺寸整除。")
        self.num_patches = (self.img_size // self.patch_size) ** 2

        self.time_embed = TimestepEmbed(self.embed_dim)

        # --- 添加类别嵌入层 ---
        self.label_embed = nn.Embedding(self.num_classes, self.embed_dim)
        # (可选) 初始化嵌入层权重
        nn.init.normal_(self.label_embed.weight, std=0.02)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,  # 使用 self.in_channels (值为4)
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

        # --- 解码器线性层: (D) -> (P*P*C=4) ---
        patch_dim = (self.patch_size ** 2) * self.in_channels  # C=4
        self.final_linear = nn.Linear(self.embed_dim, patch_dim)

        # --- 设置扩散过程参数 ---
        self.num_timesteps = model_config.get("timesteps", 1000)

        betas = get_beta_schedule(self.num_timesteps, model_config.get("schedule_type", "cosine"))
        # 确保 betas 在正确的设备上创建
        betas = betas.to(torch.float32)  # 确保类型正确

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 注册 buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # 在注册前确保 alphas_cumprod_prev 在同一设备
        alphas_cumprod_prev_init = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=torch.float32), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev_init)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas',
                             torch.sqrt(1.0 / alphas))  # 使用 .reciprocal() 可能更安全: torch.sqrt(alphas.reciprocal())
        # 在计算 posterior_variance 前确保所有张量在同一设备且类型正确
        posterior_variance_init = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', torch.clamp(posterior_variance_init, min=1e-20))  # 添加 clamp 防止除零
        posterior_mean_coef1_init = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1_init)
        posterior_mean_coef2_init = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2_init)

        self.logger.info(f"DiffusionTransformer (DiT) 模型已初始化 (输入通道: {self.in_channels}):")
        self.logger.info(f"  Embed Dim: {self.embed_dim}, Depth: {self.depth}, Heads: {self.num_heads}")
        self.logger.info(f"  Image: {self.img_size}x{self.img_size}, Patch: {self.patch_size}, Patches: {self.num_patches}")
        self.logger.info(f"  Normalization value: {self.norm_value}")

    # --- Unpatchify 辅助函数 ---
    def _unpatchify(self, x):
        """将 Token 序列 (B, N, P*P*C=4) 转换回图像 (B, C=4, H, W)."""
        P = self.patch_size
        C = self.in_channels  # C=4
        H = W = self.img_size
        num_patches_h = H // P
        num_patches_w = W // P

        x = rearrange(
            x, 'b (nh nw) (p ph c) -> b c (nh p) (nw ph)',
            c=C, p=P, ph=P, nh=num_patches_h, nw=num_patches_w
        )
        return x

    # --- 归一化/反归一化 (处理4通道) ---
    def _normalize(self, x_tensor):
        """
        将 (B, 4, H, W) 数据归一化到 [-1, 1] 范围.
        通道 0 (exists): 0 -> -1, 1 -> 1
        通道 1,2,3 (a,alpha,d): [0, norm_value] -> [-1, 1]
        """
        if x_tensor.shape[1] != 4:
            raise ValueError(f"输入张量的通道数应为4, 但收到 {x_tensor.shape[1]}")
        exists_channel = x_tensor[:, 0:1, :, :]
        other_channels = x_tensor[:, 1:, :, :]

        exists_norm = exists_channel * 2.0 - 1.0
        # 防止除零, 加上一个小的 epsilon
        other_norm = (other_channels / (self.norm_value + 1e-8)) * 2.0 - 1.0

        return torch.cat([exists_norm, other_norm], dim=1)

    def _unnormalize(self, x_tensor_norm):
        """
        将 (B, 4, H, W) 数据从 [-1, 1] 恢复到原始范围.
        通道 0 (exists): [-1, 1] -> [0, 1] (近似)
        通道 1,2,3 (a,alpha,d): [-1, 1] -> [0, norm_value]
        """
        if x_tensor_norm.shape[1] != 4:
            raise ValueError(f"输入张量的通道数应为4, 但收到 {x_tensor_norm.shape[1]}")
        exists_norm = x_tensor_norm[:, 0:1, :, :]
        other_norm = x_tensor_norm[:, 1:, :, :]

        exists_unnorm = (exists_norm + 1.0) / 2.0  # 范围 [0, 1]
        other_unnorm = ((other_norm + 1.0) / 2.0) * self.norm_value

        return torch.cat([exists_unnorm, other_unnorm], dim=1)

    # --- 前向传播 (接收标签 y) ---
    def forward(self, x_norm, t, y):
        """
        (已更新!) 接收 *归一化* 的 x_t_norm (B, 4, H, W), t (B,), 和标签 y (B,),
        预测 *归一化* 的噪声.
        """
        # 1. 时间嵌入
        t_embed = self.time_embed(t)  # (B, D)

        # 2. --- (新!) 类别嵌入 ---
        y_embed = self.label_embed(y)  # (B, D)

        # 3. --- (新!) 合并条件 ---
        c = t_embed + y_embed  # (B, D) (简单相加)

        # 4. Patch 嵌入 + 位置编码 (不变)
        x = self.patch_embed(x_norm)
        x = self.pos_embed(x)

        # 5. Transformer 核心 (传入合并后的 c)
        for block in self.blocks:
            x = block(x, c)  # <-- 使用合并的 c

        # 6. 解码 (不变)
        x = self.final_norm(x)
        x = self.final_linear(x)
        noise_pred_norm = self._unpatchify(x)

        return noise_pred_norm

    # --- 扩散过程辅助函数 ---
    def _get_tensor_values(self, val, t):
        """
        从调度表 `val` (shape [T]) 中, 根据 `t` (shape [B]) 提取正确的时间步参数.
        并重塑为 (B, 1, 1, 1) 以便广播.
        """
        batch_size = t.shape[0]
        # 确保 t 是 LongTensor 并且在 val 的设备上
        t_long = t.long().to(val.device)
        # 使用 t 作为索引, 从 val 中 "收集" 正确的参数
        out = val.gather(-1, t_long)
        # 将 (B,) 重塑为 (B, 1, 1, 1)
        return out.reshape(batch_size, 1, 1, 1)

    def q_sample(self, x_start_norm, t, noise=None):
        """
        (修正后版本) 前向加噪过程.
        接收 *归一化* x_start_norm, 返回 *归一化* 的 x_t_norm.
        """
        if noise is None:
            noise = torch.randn_like(x_start_norm)

        sqrt_alphas_cumprod_t = self._get_tensor_values(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)

        # x_t_norm = sqrt(alpha_bar_t) * x_start_norm + sqrt(1 - alpha_bar_t) * noise_norm
        x_t_norm = sqrt_alphas_cumprod_t * x_start_norm + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t_norm

    def _x0_from_noise(self, x_t_norm, t, noise_norm):
        """
        从 *归一化* 的 x_t 和预测的 *归一化* 噪声中计算 *归一化* 的 x_0_pred_norm
        """
        # 使用 .reciprocal() 可能更安全
        sqrt_recip_alphas_cumprod_t = self._get_tensor_values(self.alphas_cumprod.sqrt().reciprocal(), t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)

        x_0_pred_norm = sqrt_recip_alphas_cumprod_t * (x_t_norm - sqrt_one_minus_alphas_cumprod_t * noise_norm)
        return x_0_pred_norm

    # --- 反向去噪 (p_sample, 无中间钳位) ---
    def p_sample(self, x_t_norm, t, t_index, y, guidance_fn=None, guidance_scale=1.0):
        """
        (已更新!) 反向去噪一步, 支持RL引导, **接收标签 y**.
        输入: 归一化的 x_t_norm, t, y
        输出: 归一化的 x_{t-1}_norm
        """
        # 1. 获取扩散参数
        betas_t = self._get_tensor_values(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = self._get_tensor_values(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = self._get_tensor_values(self.sqrt_recip_alphas, t)

        # 2. --- 预测噪声 (传入 y) ---
        predicted_noise_norm = self.forward(x_t_norm, t, y)  # <-- 传入

        # 3. (可选) RL 引导 (RL Agent 的意见)
        if guidance_fn is not None:
            gradient = guidance_fn(x_t_norm, t)
            # 应用引导 (与之前相同)
            predicted_noise_norm = predicted_noise_norm - sqrt_one_minus_alphas_cumprod_t * guidance_scale * gradient

        # 4. 计算 x_0 的初步估计 (归一化)
        #    *** 使用可能被引导过的噪声 ***
        x_0_pred_norm = self._x0_from_noise(x_t_norm, t, predicted_noise_norm)

        # --- 5. (!!!) 添加钳位 (Clamp) (!!!) ---
        #    这是修复不稳定的关键步骤。
        #    我们将估算的 x_0 强制限制在 [-1, 1] 范围内。
        x_0_pred_norm = torch.clamp(x_0_pred_norm, -1.0, 1.0)  # 仅对异常数据进行处理
        # (可选)把所有数据强制归一化到[-1,1]，但是会扭曲数据
        # x_0_pred_norm = torch.tanh(x_0_pred_norm)

        # --- 6. 使用【原始预测】的 x_0_pred_norm 计算 x_{t-1} 的均值 ---
        posterior_mean_coef1_t = self._get_tensor_values(self.posterior_mean_coef1, t)
        posterior_mean_coef2_t = self._get_tensor_values(self.posterior_mean_coef2, t)
        # *** 使用未经钳位的 x_0_pred_norm ***
        posterior_mean = posterior_mean_coef1_t * x_0_pred_norm + posterior_mean_coef2_t * x_t_norm

        # 7. 添加噪声 (采样)
        posterior_variance_t = self._get_tensor_values(self.posterior_variance, t)
        noise = torch.randn_like(x_t_norm)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t_norm.shape) - 1)))

        # 返回归一化的 x_{t-1}_norm
        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    # --- 采样函数 (sample, 在末尾钳位) ---
    @torch.no_grad()
    def sample(self, num_samples, y, guidance_fn=None, guidance_scale=1.0):
        """
        (已更新!) 完整的采样循环. **接收标签 y**.
        在采样结束后进行反归一化和钳位.
        返回: *钳位后* 且 *未归一化* 的 Numpy 数组列表 [(H, W, 4), ...]
        """
        self.logger.info(f"开始采样 {num_samples} 个新机构 (格式: [exists, a, alpha, d])...")
        device = next(self.parameters()).device

        # 1. 从纯噪声 x_T 开始 (归一化空间)
        shape = (num_samples, self.in_channels, self.img_size, self.img_size)
        x_t_norm = torch.randn(shape, device=device)

        # 2. 从 T 循环到 1 (传入 y)
        for t_index in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM 采样", total=self.num_timesteps):
            t = torch.full((num_samples,), t_index, device=device, dtype=torch.long)
            # --- (核心修改!) 调用 p_sample 时传入 y ---
            x_t_norm = self.p_sample(x_t_norm, t, t_index, y, guidance_fn, guidance_scale)

        # 3. x_0_norm 就是最终的 x_t_norm (仍然在 [-1, 1] 范围)
        x_0_norm = x_t_norm

        # --- 4. 在采样结束后进行【一次性】反归一化和钳位 ---
        # 4a. 反归一化
        x_0_unnorm = self._unnormalize(x_0_norm) # (B, 4, H, W), 物理范围 (近似)

        # 4b. 分离通道
        exists_unnorm, a_unnorm, alpha_unnorm, d_unnorm = torch.chunk(x_0_unnorm, 4, dim=1)

        # 4c. 钳位 exists 通道 (阈值 0.5)
        print(exists_unnorm)
        exists_clamped = (exists_unnorm > 0.5).float() # 变为 0.0 或 1.0
        print(exists_clamped)

        # 4d. 钳位 a, alpha, d 通道
        a_clamped = torch.clamp(a_unnorm, min=0.0)
        alpha_clamped = torch.clamp(alpha_unnorm, min=0.0, max=math.pi / 2.0)
        d_clamped = torch.clamp(d_unnorm, min=0.0)

        # 4e. 如果 exists=0, 强制 a,alpha,d 为 0
        a_clamped = a_clamped * exists_clamped
        alpha_clamped = alpha_clamped * exists_clamped
        d_clamped = d_clamped * exists_clamped

        # 4f. 重新组合通道 (最终的、钳位后的、未归一化的结果)
        x_0_final_unnorm = torch.cat([exists_clamped, a_clamped, alpha_clamped, d_clamped], dim=1)

        # 5. 转换 (B, C, H, W) -> (B, H, W, C) 并移到 CPU
        x_0_numpy = x_0_final_unnorm.permute(0, 2, 3, 1).cpu().numpy()

        # 6. 分割成列表
        generated_mechanisms = [x_0_numpy[i] for i in range(num_samples)]

        self.logger.info(f"采样完成. 共生成 {len(generated_mechanisms)} 个机构张量 (已应用最终钳位)。")
        return generated_mechanisms # 返回钳位后、未归一化的 (H, W, 4) Numpy 数组列表