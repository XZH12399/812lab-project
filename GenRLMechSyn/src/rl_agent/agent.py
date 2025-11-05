# src/rl_agent/agent.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging  # 导入 logging

# 我们复用 DiT 的 "零件"
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
    它的工作是:
    - 输入: x_t (归一化), t (时间), y (标签)
    - 输出: 预测这个 (x_t, t, y) 最终会得到的 *总奖励*
    """

    def __init__(self, config):
        super().__init__()
        self.logger = logging.getLogger()  # 获取 logger
        dit_config = config['diffusion_model']

        self.img_size = dit_config.get("img_size", 30)
        self.patch_size = dit_config.get("patch_size", 5)

        # --- 确保 in_channels 与模型一致 ---
        self.in_channels = dit_config.get("in_channels", 4)
        if self.in_channels != 4:
            self.logger.warning(f"RLAgent: in_channels 配置不是 4, 而是 {self.in_channels}")

        self.embed_dim = dit_config.get("embed_dim", 768)

        # --- 读取类别数量 ---
        self.num_classes = dit_config.get("num_classes", 1)
        if self.num_classes <= 0: raise ValueError("num_classes 必须大于 0")

        # (从 config 读取深度, 如果存在)
        self.rl_depth = config.get('rl_agent', {}).get('depth', 4)

        # --- 组装奖励预测网络 R(x_t, t, y) ---
        self.time_embed = TimestepEmbed(self.embed_dim)

        # --- 添加类别嵌入层 (与 DiT 保持一致) ---
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
        (已更新!) 前向传播: 预测 x_t, t, y 的预期奖励.
        x_t (B, C, H, W) - 归一化的嘈杂机构
        t (B,)         - 整数时间步
        y (B,)         - 标签索引
        """
        # 1. 时间嵌入
        t_embed = self.time_embed(t)

        # 2. 类别嵌入
        y_embed = self.label_embed(y)

        # 3. 合并条件
        c = t_embed + y_embed

        # 4. Patch 嵌入 + 位置编码
        x = self.patch_embed(x_t_norm)
        x = self.pos_embed(x)

        # 5. Transformer 核心 (传入合并后的 c)
        for block in self.blocks:
            x = block(x, c)

        # 6. 预测 "头"
        x = self.final_norm(x)
        x = x.mean(dim=1)  # (B, N, D) -> (B, D) (全局平均池化)
        predicted_score = self.head(x)  # (B, D) -> (B, 1)

        return predicted_score.squeeze(1)  # (B,)

    def get_guidance_fn(self, guidance_scale=1.0):
        """
        返回一个闭包函数 (closure), 该函数接收 (x_t, t, y).
        """

        def guidance_fn(x_t_norm, t, y):  # <-- 接收 y
            """
            计算引导梯度: grad_x_t( R(x_t, t, y) )
            """
            with torch.enable_grad():
                x_t_grad = x_t_norm.detach().clone().requires_grad_(True)

                # 1. 运行奖励模型, 预测分数 (传入 y)
                predicted_score = self.forward(x_t_grad, t, y)  # <-- 传入 y

                # 2. 计算梯度: d(Score) / d(x_t)
                grad = torch.autograd.grad(
                    outputs=predicted_score.sum(),
                    inputs=x_t_grad,
                    create_graph=False,
                    retain_graph=False
                )[0]

            # 3. 返回缩放后的梯度
            return grad * guidance_scale

        return guidance_fn  # 返回的函数现在需要 (x_t, t, y)

    def update_policy(self, experiences, diffusion_model, optimizer, device):
        """
        训练奖励预测模型 (RLAgent).
        experiences: 包含 (x_0_tensor_NORMALIZED, final_score, label_idx)
        diffusion_model: 我们的 DiT 模型 (用于 q_sample)
        """
        if not experiences:
            self.logger.info("Replay Buffer 为空, 跳过 RL 训练。")
            return None

        # --- 提取所有 3 个元素 ---
        x_0_norm_tensors = [exp[0].to(device) for exp in experiences]
        scores = [exp[1] for exp in experiences]
        y_labels = [exp[2].to(device) for exp in experiences]  # 提取标签

        dataset = TensorDataset(
            torch.stack(x_0_norm_tensors),
            torch.tensor(scores, dtype=torch.float32, device=device),
            torch.stack(y_labels)  # <-- 添加标签到数据集中
        )
        # 增加 batch_size 以提高训练稳定性 (如果显存允许)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
        diffusion_model.eval()
        total_loss = 0

        # 2. 训练循环 (为 RLAgent 训练 5 个 epoch)
        for _ in range(5):
            # --- 加载 x_0_norm, target_scores, y ---
            for x_0_norm, target_scores, y in loader:
                optimizer.zero_grad()

                # 2a. 随机采样一个时间步 t
                t = torch.randint(
                    0,
                    diffusion_model.num_timesteps,
                    (x_0_norm.size(0),),
                    device=device
                ).long()

                # 2b. 创建 "历史照片" x_t (加噪)
                noise = torch.randn_like(x_0_norm)
                # q_sample 接收 *归一化* 输入
                x_t_norm = diffusion_model.q_sample(x_0_norm, t, noise)

                # 2c. RLAgent ("先知") 做出预测 (传入 y)
                predicted_scores = self.forward(x_t_norm, t, y)  # <-- 传入 y

                # 3. 计算损失 (预测分数 vs 真实分数)
                loss = F.mse_loss(predicted_scores, target_scores)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if len(loader) == 0:
            return 0.0  # 避免除零

        avg_loss = total_loss / (len(loader) * 5)
        # (移除 print, 因为 pipeline 会打印)
        return avg_loss  # <-- 返回 avg_loss