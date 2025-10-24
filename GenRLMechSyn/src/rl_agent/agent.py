# src/rl_agent/agent.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# 我们复用 DiT 的 "零件"
try:
    from ..diffusion_model.modules import (
        PatchEmbed,
        PositionalEncoding,
        BasicTransformerBlock,
        TimestepEmbed  # R(x_t, t) 版本
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
    强化学习智能体.
    这是一个 "奖励预测模型" (或 Q-Function / Value-Function).
    它的工作是:
    - 输入: 一个 *嘈杂* 的机构张量 x_t 和时间 t
    - 输出: 预测这个 x_t 最终会得到的 *总奖励*
    """

    def __init__(self, config):
        super().__init__()
        dit_config = config['diffusion_model']

        self.img_size = dit_config.get("img_size", 30)
        self.patch_size = dit_config.get("patch_size", 5)
        self.in_channels = dit_config.get("in_channels", 3)
        self.embed_dim = dit_config.get("embed_dim", 768)

        # 我们使用一个 "更浅" 的 DiT 作为我们的奖励模型
        self.rl_depth = 4  # (e.g., 4 层, 而不是 12 层)

        # --- 组装奖励预测网络 R(x_t, t) ---
        self.time_embed = TimestepEmbed(self.embed_dim)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
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
        # (B, N, D) -> (B, D) -> (B, 1)
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, 1)  # 预测一个标量 (分数)
        )

        print(f"--- 强化学习 (RL) 智能体已初始化 (奖励预测模型) ---")
        print(f"  RL Model Depth: {self.rl_depth}")

    def forward(self, x_t, t):
        """
        前向传播: 预测 x_t 在时间 t 的预期奖励.
        x_t (B, C, H, W) - 嘈杂的机构
        t (B,)         - 整数时间步
        """
        # 1. 时间嵌入
        c = self.time_embed(t)

        # 2. Patch 嵌入
        x = self.patch_embed(x_t)
        x = self.pos_embed(x)

        # 3. Transformer 核心
        for block in self.blocks:
            x = block(x, c)

        # 4. 预测 "头"
        x = self.final_norm(x)
        x = x.mean(dim=1)  # (B, N, D) -> (B, D) (全局平均池化)
        predicted_score = self.head(x)  # (B, D) -> (B, 1)

        return predicted_score.squeeze(1)  # (B,)

    def get_guidance_fn(self, guidance_scale=1.0):
        """
        返回一个闭包函数 (closure), 该函数将被 DiT 的采样器调用.
        """

        def guidance_fn(x_t, t):
            """
            计算引导梯度: grad_x_t( R(x_t, t) )
            """
            # 我们需要计算梯度, 所以打开 grad
            with torch.enable_grad():
                # 复制 x_t, 并告诉 PyTorch 我们需要计算它对 x_t 的梯度
                x_t_grad = x_t.detach().clone().requires_grad_(True)

                # 1. 运行奖励模型, 预测分数
                # (B, C, H, W) -> (B,)
                predicted_score = self.forward(x_t_grad, t)

                # 2. 计算梯度: d(Score) / d(x_t)
                #    我们对 .sum() 求导, 以便在批次中独立计算
                grad = torch.autograd.grad(
                    outputs=predicted_score.sum(),
                    inputs=x_t_grad,
                    create_graph=False,
                    retain_graph=False
                )[0]

            # 3. 返回缩放后的梯度
            return grad * guidance_scale

        return guidance_fn

    def update_policy(self, experiences, diffusion_model, optimizer, device):
        """
        训练奖励预测模型 (RLAgent).
        experiences: 一个列表, 包含 (x_0_tensor_NORMALIZED, final_score)
        diffusion_model: 我们的 DiT 模型 (用于 q_sample)
        """

        # 1. 准备数据
        if not experiences:
            print("Replay Buffer 为空, 跳过 RL 训练。")
            return

        # experiences 已经是 (tensor_norm, score)
        x_0_norm_tensors = [exp[0].to(device) for exp in experiences]
        scores = [exp[1] for exp in experiences]

        dataset = TensorDataset(
            torch.stack(x_0_norm_tensors),
            torch.tensor(scores, dtype=torch.float32, device=device)
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.train()
        diffusion_model.eval()

        total_loss = 0

        # 2. 训练循环 (为 RLAgent 训练 5 个 epoch)
        for _ in range(5):
            for x_0_norm, target_scores in loader:  # x_0_norm 是 [B, C, H, W] (已归一化)
                optimizer.zero_grad()

                # 2a. 随机采样一个时间步 t (e.g., t=500)
                t = torch.randint(
                    0,
                    diffusion_model.num_timesteps,
                    (x_0_norm.size(0),),
                    device=device
                ).long()

                # 2b. 创建 "历史照片" x_t (加噪)
                noise = torch.randn_like(x_0_norm)
                x_t = diffusion_model.q_sample(x_0_norm, t, noise)

                # 2c. RLAgent ("先知") 做出预测
                predicted_scores = self.forward(x_t, t)

                # 3. 计算损失 (预测分数 vs 真实分数)
                loss = F.mse_loss(predicted_scores, target_scores)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(loader) * 5)
        print(f"--- RL Agent 训练完成. 平均 Loss: {avg_loss:.6f} ---")