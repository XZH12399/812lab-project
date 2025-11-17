# src/rl_agent/agent.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging

# ( ... 导入 ... )
try:
    from ..diffusion_model.modules import (
        PatchEmbed,
        PositionalEncoding,
        BasicTransformerBlock,
        TimestepEmbed
    )
except ImportError:
    from src.diffusion_model.modules import (
        PatchEmbed,
        PositionalEncoding,
        BasicTransformerBlock,
        TimestepEmbed
    )


class RLAgent(nn.Module):
    """
    强化学习智能体 (奖励预测模型).
    输入: x_t (归一化, 4通道), t (时间), y (标签)
    输出: 预测这个 (x_t, t, y) 最终会得到的 *总奖励*
    """

    def __init__(self, config):
        super().__init__()
        self.logger = logging.getLogger()
        dit_config = config['diffusion_model']

        self.img_size = dit_config.get("img_size", 30)
        self.patch_size = dit_config.get("patch_size", 5)

        self.in_channels = dit_config.get("in_channels", 4)  # 默认 4
        if self.in_channels != 4:
            self.logger.warning(f"RLAgent: in_channels 配置不是 4, 而是 {self.in_channels}")

        self.embed_dim = dit_config.get("embed_dim", 768)
        self.num_classes = dit_config.get("num_classes", 1)
        if self.num_classes <= 0: raise ValueError("num_classes 必须大于 0")
        self.rl_depth = config.get('rl_agent', {}).get('depth', 4)

        # --- 组装奖励预测网络 R(x_t, t, y) ---
        self.time_embed = TimestepEmbed(self.embed_dim)
        self.label_embed = nn.Embedding(self.num_classes, self.embed_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,  # <-- 使用 4
            embed_dim=self.embed_dim
        )
        self.pos_embed = PositionalEncoding(
            num_patches=(self.img_size // self.patch_size) ** 2,
            embed_dim=self.embed_dim
        )
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=self.embed_dim, num_heads=dit_config.get("num_heads", 12))
            for _ in range(self.rl_depth)
        ])

        # --- 最终的 "头" (Head) ---
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, 1)  # 预测一个标量 (分数)
        )

        self.logger.info(f"--- 强化学习 (RL) 智能体已初始化 (奖励预测模型) ---")
        self.logger.info(f"  RL Model Depth: {self.rl_depth}")

    def forward(self, x_t_norm, t, y):
        """
        前向传播: 预测 x_t, t, y 的预期奖励.
        x_t (B, C=4, H, W) - 归一化的嘈杂机构
        """
        t_embed = self.time_embed(t)
        y_embed = self.label_embed(y)
        c = t_embed + y_embed
        x = self.patch_embed(x_t_norm)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        predicted_score = self.head(x)
        return predicted_score.squeeze(1)

    def get_guidance_fn(self, guidance_scale=1.0):

        def guidance_fn(x_t_norm, t, y):
            with torch.enable_grad():
                x_t_grad = x_t_norm.detach().clone().requires_grad_(True)
                predicted_score = self.forward(x_t_grad, t, y)
                grad = torch.autograd.grad(
                    outputs=predicted_score.sum(),
                    inputs=x_t_grad,
                    create_graph=False,
                    retain_graph=False
                )[0]
            return grad * guidance_scale

        return guidance_fn

    def update_policy(self, experiences, diffusion_model, optimizer, device):
        if not experiences:
            self.logger.info("Replay Buffer 为空, 跳过 RL 训练。")
            return None

        # --- 提取所有 3 个元素 ---
        x_0_norm_tensors = [exp[0].to(device) for exp in experiences]  # (C=4, H, W)
        scores = [exp[1] for exp in experiences]
        y_labels = [exp[2].to(device) for exp in experiences]

        dataset = TensorDataset(
            torch.stack(x_0_norm_tensors),
            torch.tensor(scores, dtype=torch.float32, device=device),
            torch.stack(y_labels)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
        # diffusion_model.eval()
        total_loss = 0

        # 2. 训练循环 (为 RLAgent 训练 5 个 epoch)
        for _ in range(5):
            for x_0_norm, target_scores, y in loader:  # x_0_norm (B, 4, H, W)
                optimizer.zero_grad()
                t = torch.randint(
                    0,
                    diffusion_model.num_timesteps,
                    (x_0_norm.size(0),),
                    device=device
                ).long()

                # 2b. 创建 "历史照片" x_t (加噪)
                noise = torch.randn_like(x_0_norm)  # (B, 4, H, W)
                x_t_norm = diffusion_model.q_sample(x_0_norm, t, noise)  # (B, 4, H, W)

                # 2c. RLAgent ("先知") 做出预测 (传入 y)
                predicted_scores = self.forward(x_t_norm, t, y)

                # 3. 计算损失 (预测分数 vs 真实分数)
                loss = F.mse_loss(predicted_scores, target_scores)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if len(loader) == 0:
            return 0.0

        avg_loss = total_loss / (len(loader) * 5)
        return avg_loss