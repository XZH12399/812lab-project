# src/pipeline.py

import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 导入所有模块
from .utils import dataloader
from .evaluator.evaluator import MechanismEvaluator
from .diffusion_model.model import DiffusionModel
from .rl_agent.agent import RLAgent # 现在这是一个 nn.Module

class TrainingPipeline:
    def __init__(self, config, project_root):
        self.config = config
        self.project_root = project_root

        # --- 1. 初始化硬件 ---
        self.device = self._setup_device()

        # --- 2. 加载数据归一化参数和生成参数 ---
        self.norm_value = config['data']['normalization_value']
        self.guidance_scale = config['generation'].get('rl_guidance_scale', 1.0)

        # --- 3. 初始化所有模块 ---
        print("--- 正在初始化所有模块... ---")
        self.diffusion_model = DiffusionModel(config).to(self.device) # 传递整个 config

        # --- (新!) 加载检查点 (如果指定) ---
        load_path = config['training'].get('load_checkpoint_path')
        if load_path and load_path.lower() != 'null':
            checkpoint_path = os.path.join(project_root, load_path)  # 构建绝对路径
            if os.path.exists(checkpoint_path):
                try:
                    print(f"--- 正在从检查点加载 DiT 模型权重 ---")
                    print(f"路径: {checkpoint_path}")
                    # 加载 state_dict (只加载权重)
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    # 严格模式(strict=True)确保加载的键与模型完全匹配
                    self.diffusion_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print("--- DiT 模型权重加载成功 ---")
                except Exception as e:
                    print(f"[警告] 加载检查点失败: {e}")
                    print("将使用随机初始化的权重。")
            else:
                print(f"[警告] 指定的检查点文件不存在: {checkpoint_path}")
                print("将使用随机初始化的权重。")
        else:
            print("--- 未指定加载检查点, DiT 模型使用随机初始化权重 ---")

        self.evaluator = MechanismEvaluator(config) # 传递整个 config
        self.rl_agent = RLAgent(config).to(self.device) # 传递整个 config

        # --- 4. 设置优化器 (现在有两个!) ---
        self.dit_optimizer = optim.AdamW(
            self.diffusion_model.parameters(),
            lr=float(config['training'].get('learning_rate', 0.0001)) # 确保是 float
        )
        self.rl_optimizer = optim.AdamW(
            self.rl_agent.parameters(),
            lr=float(config['training'].get('rl_learning_rate', 0.00001)) # 确保是 float
        )

        # --- 5. 加载实验开关 ---
        self.enable_rl_guidance = config['training'].get('enable_rl_guidance', True)
        self.enable_augmentation = config['training'].get('enable_augmentation', True)
        self.acceptance_threshold = config['generation'].get('acceptance_threshold', -float('inf'))

        # --- 6. 初始化经验库 (Replay Buffer) ---
        self.replay_buffer = self.load_replay_buffer()

        print("--- 训练流程已准备就绪 ---")
        print(f"  RL 引导: {self.enable_rl_guidance}")
        print(f"  数据增强: {self.enable_augmentation}")
        if self.enable_augmentation:
            print(f"  增强阈值 (分数 >=): {self.acceptance_threshold}")

    def _setup_device(self):
        """设置训练设备 (GPU 或 CPU)"""
        if self.config['training'].get('device', 'cpu') == 'cuda' and torch.cuda.is_available():
            print("训练将在 NVIDIA CUDA GPU 上运行。")
            return torch.device("cuda")
        else:
            print("训练将在 CPU 上运行。")
            return torch.device("cpu")

    # --- 归一化/反归一化 辅助函数 ---
    # (模型内部有自己的版本, 但 pipeline 在处理 Replay Buffer 时也需要)
    def _normalize(self, x_tensor):
        """(新!) 将 (B, 4, H, W) 或 (C, H, W) 数据归一化到 [-1, 1] 范围."""
        if x_tensor.dim() == 3: # (C, H, W) -> (1, C, H, W)
             x_tensor = x_tensor.unsqueeze(0)
             was_3d = True
        else:
             was_3d = False

        exists_channel = x_tensor[:, 0:1, :, :]
        other_channels = x_tensor[:, 1:, :, :]
        exists_norm = exists_channel * 2.0 - 1.0
        other_norm = (other_channels / (self.norm_value + 1e-8)) * 2.0 - 1.0
        result = torch.cat([exists_norm, other_norm], dim=1)

        if was_3d:
            return result.squeeze(0) # (1, C, H, W) -> (C, H, W)
        return result

    def _unnormalize(self, x_tensor_norm):
        """(新!) 将 (B, 4, H, W) 或 (C, H, W) 数据从 [-1, 1] 恢复."""
        if x_tensor_norm.dim() == 3: # (C, H, W) -> (1, C, H, W)
             x_tensor_norm = x_tensor_norm.unsqueeze(0)
             was_3d = True
        else:
             was_3d = False

        exists_norm = x_tensor_norm[:, 0:1, :, :]
        other_norm = x_tensor_norm[:, 1:, :, :]
        exists_unnorm = (exists_norm + 1.0) / 2.0
        other_unnorm = ((other_norm + 1.0) / 2.0) * self.norm_value
        result = torch.cat([exists_unnorm, other_unnorm], dim=1)

        if was_3d:
            return result.squeeze(0) # (1, C, H, W) -> (C, H, W)
        return result


    def load_replay_buffer(self):
        """
        加载现有的增强数据集 (作为 RL Agent 的训练数据).
        存储 *归一化* 的张量。
        """
        print("正在加载 Replay Buffer (来自 augmented_dataset)...")
        manifest_path = self.config['data']['augmented_manifest_path']
        data_dir = os.path.dirname(manifest_path)

        if not os.path.exists(manifest_path):
            print("信息: augmented_manifest.json 不存在, Replay Buffer 为空。")
            return []

        with open(manifest_path, 'r', encoding='utf-8') as f: # 添加 encoding
            manifest = json.load(f)

        experiences = []
        for entry in manifest:
            npz_path = os.path.join(data_dir, entry['data_path'])
            try:
                with np.load(npz_path) as data:
                    # 加载未归一化的 (H, W, 4) Numpy
                    tensor_unnorm_numpy = data['mechanism_tensor']
                    # (H,W,C) -> (C,H,W) Torch, 未归一化
                    tensor_unnorm_torch = torch.from_numpy(np.transpose(tensor_unnorm_numpy, (2, 0, 1))).float()
                    # --- 进行归一化 ---
                    tensor_norm = self._normalize(tensor_unnorm_torch)

                    score = entry['metadata']['score']
                    # 存储归一化的 Torch 张量
                    experiences.append((tensor_norm, score))
            except Exception as e:
                print(f"警告: 无法加载 Replay Buffer 条目 {npz_path}. 原因: {e}")

        print(f"Replay Buffer 加载完成. 共 {len(experiences)} 个历史经验。")
        return experiences

    def _generate_and_augment(self, cycle_num):
        """
        步骤 1 & 2: 生成 (带引导), 评估, 并扩充数据集.
        返回: 新的经验 [(tensor_NORMALIZED, score), ...], 用于训练 RL Agent.
        """
        print("\n--- 步骤 1 & 2: 生成, 评估, 扩充 ---")

        self.diffusion_model.eval()
        self.rl_agent.eval()

        # 1. 获取 RL "引导函数" (如果启用)
        guidance_fn = None
        if self.enable_rl_guidance:
            print("RL 引导已启用。")
            guidance_fn = self.rl_agent.get_guidance_fn(self.guidance_scale)
        else:
            print("RL 引导已关闭。将使用纯 DiT 采样。")

        # 2. 生成新机构 (model.sample 返回 *未归一化* 的 Numpy 数组列表)
        new_mech_tensors_unnorm_numpy = self.diffusion_model.sample(
            num_samples=self.config['generation']['num_to_generate'],
            guidance_fn=guidance_fn,
            guidance_scale=self.guidance_scale
        )

        # 3. 评估与筛选
        print(f"评估 {len(new_mech_tensors_unnorm_numpy)} 个新生成的机构...")
        new_experiences_for_rl = []
        good_mechanisms_to_save = []

        for tensor_unnorm_numpy in new_mech_tensors_unnorm_numpy:
            # 评估器接收 (30, 30, 4) 未归一化 Numpy 张量, 返回总奖励
            score = self.evaluator.evaluate(tensor_unnorm_numpy)
            print(f"  Generated sample score: {score:.4f}")

            # --- 准备给 RL 的数据 (归一化, C,H,W, Torch) ---
            # (H,W,C) -> (C,H,W) Torch, 未归一化
            tensor_torch_unnorm = torch.from_numpy(np.transpose(tensor_unnorm_numpy, (2, 0, 1))).float()
            # 归一化
            tensor_torch_norm = self._normalize(tensor_torch_unnorm)

            # 添加 *归一化* 的张量和分数到经验列表
            # 将张量移动到设备以备 RL 训练
            new_experiences_for_rl.append((tensor_torch_norm.to(self.device), score))

            # 检查是否满足保存条件
            if self.enable_augmentation and score >= self.acceptance_threshold:
                 new_entry = {
                     "tensor": tensor_unnorm_numpy, # 保存 (30, 30, 4) 未归一化 Numpy
                     "metadata": {
                         "source": "generated",
                         "generation_cycle": cycle_num + 1,
                         "score": score
                     }
                 }
                 good_mechanisms_to_save.append(new_entry)

        print(f"评估完成. {len(good_mechanisms_to_save)} / {len(new_mech_tensors_unnorm_numpy)} 个机构满足保存要求 (分数 >= {self.acceptance_threshold})。")

        # 4. 扩充数据集 (保存到磁盘, 如果启用)
        if self.enable_augmentation:
            if good_mechanisms_to_save:
                print(f"数据增强已启用。正在保存 {len(good_mechanisms_to_save)} 个机构...")
                # dataloader 需要更新以处理新格式
                dataloader.add_mechanisms_to_dataset(
                    good_mechanisms_to_save,
                    self.config['data']['augmented_manifest_path']
                )
            else:
                print("数据增强已启用, 但没有合格的机构可保存。")
        else:
            print("数据增强已关闭。跳过保存。")

        return new_experiences_for_rl

    def _train_rl_agent(self, new_experiences):
        """
        步骤 3a: 训练 RL Agent (奖励预测模型).
        """
        # 如果 RL 引导未启用, 我们也不训练 RLAgent
        if not self.enable_rl_guidance:
             print("\n--- RL 引导已关闭, 跳过 RL 智能体训练 ---")
             return

        print("\n--- 步骤 3a: 训练 RL 智能体 ---")

        # 1. 将新经验添加到 Replay Buffer (经验已经是归一化的)
        # (可选: 限制 buffer 大小以避免内存问题)
        buffer_limit = self.config['training'].get('replay_buffer_limit', 50000)
        self.replay_buffer.extend(new_experiences)
        if len(self.replay_buffer) > buffer_limit:
            self.replay_buffer = self.replay_buffer[-buffer_limit:] # 保留最新的 N 个

        print(f"Replay Buffer 中共有 {len(self.replay_buffer)} 个经验。")

        if not self.replay_buffer:
            print("Replay Buffer 为空, 跳过 RL 训练。")
            return

        # 2. 训练 RL Agent (传递归一化的 buffer)
        self.rl_agent.update_policy(
            self.replay_buffer, # 包含归一化的张量
            self.diffusion_model,
            self.rl_optimizer,
            self.device
        )

    def _train_dit_model(self, cycle_num):
        """
        步骤 3b: 训练 DiT 模型 (模仿者).
        (已更新, 支持 enable_augmentation 开关)
        """
        # --- 1. 确定要加载的数据 ---
        initial_manifest_path = self.config['data']['initial_manifest_path']
        augmented_manifest_path = None

        if self.enable_augmentation:
            print("数据增强已启用。DiT 将在 [初始 + 增强] 数据集上训练。")
            augmented_manifest_path = self.config['data']['augmented_manifest_path']
        else:
            print("数据增强已关闭。DiT 将仅在 [初始] 数据集上训练。")

        # --- 2. 加载数据 ---
        current_dataloader = dataloader.get_dataloader(
            initial_manifest_path,
            augmented_manifest_path,
            self.config['training']['batch_size'],
            shuffle=True
        )

        # --- 3. 检查数据加载结果 ---
        if current_dataloader is None or len(current_dataloader.dataset) == 0:
            print("错误：数据加载失败或数据集为空, 跳过 DiT 训练。")
            return

        print(f"\n--- 步骤 3b: 训练 DiT (模仿者) ---")
        print(f"将使用 {len(current_dataloader.dataset)} 个机构进行训练。")

        # --- 4. 训练循环 ---
        self.diffusion_model.train() # 设置为训练模式

        num_epochs = self.config['training']['epochs_per_cycle']
        for epoch in range(num_epochs):
            total_loss = 0.0
            # dataloader 返回未归一化的 (B, C, H, W) torch 张量
            for x_start in current_dataloader:
                x_start = x_start.to(self.device)
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()

                # 随机采样时间步
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                # q_sample 接收未归一化 x_start, 返回归一化 x_t_norm
                # noise 用于比较, 形状与 x_start 相同, 但值是标准高斯
                noise = torch.randn_like(x_start) # (B, 4, H, W), 标准高斯
                x_t_norm = self.diffusion_model.q_sample(x_start, t, noise)

                # forward 接收归一化 x_t, 预测归一化 noise
                predicted_noise_norm = self.diffusion_model(x_t_norm, t)

                # 损失在归一化空间中计算
                # 比较预测的归一化噪声 和 用于生成 x_t_norm 的标准高斯噪声
                loss = F.mse_loss(predicted_noise_norm, noise)

                loss.backward()
                self.dit_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  DiT Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    def run(self):
        """
        执行完整的三步循环：生成 -> 训练RL -> 训练DiT
        """
        num_cycles = self.config['training']['num_cycles']
        print(f"\n--- 开始总共 {num_cycles} 轮的训练循环 ---")

        for cycle in range(num_cycles):
            print(f"\n===== [ 完整循环 {cycle + 1}/{num_cycles} ] =====")

            # 步骤 1 & 2: 生成, 评估, 扩充
            # (返回归一化的新经验)
            new_experiences = self._generate_and_augment(cycle)

            # 步骤 3a: 训练 RL 智能体 (如果启用)
            self._train_rl_agent(new_experiences)

            # 步骤 3b: 训练 DiT 模型
            self._train_dit_model(cycle)

            # (可选) 在每个循环后保存模型检查点
            # torch.save(...)

        print("\n===== 所有训练循环完成! =====")

        # --- (新!) 在训练结束后保存最终模型 ---
        self.save_checkpoint("final")  # 调用保存函数

    # --- (新!) 添加保存检查点的函数 ---
    def save_checkpoint(self, identifier="final"):
        """保存 DiT 模型的权重"""
        save_path_config = self.config['training'].get('save_checkpoint_path')
        if not save_path_config or save_path_config.lower() == 'null':
            print("--- 未指定保存路径, 跳过保存检查点 ---")
            return

        # 构建文件名 (例如 checkpoints/dit_model_cycle10.pth 或 checkpoints/dit_model_final.pth)
        base_path = os.path.join(self.project_root, save_path_config)
        save_dir = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        # final_filename = f"{name}_{identifier}{ext}" # 可以添加 cycle 编号
        final_filename = filename  # 或者直接使用配置文件中的名字

        final_save_path = os.path.join(save_dir, final_filename)

        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        try:
            print(f"\n--- 正在保存 DiT 模型检查点 ---")
            print(f"路径: {final_save_path}")
            torch.save({
                'model_state_dict': self.diffusion_model.state_dict(),
                # (可选) 保存优化器状态
                # 'optimizer_state_dict': self.dit_optimizer.state_dict(),
                # (可选) 保存其他信息, 如 cycle 数
                # 'cycle': identifier
            }, final_save_path)
            print("--- 模型检查点保存成功 ---")
        except Exception as e:
            print(f"[错误] 保存检查点失败: {e}")