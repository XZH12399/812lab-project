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
from .rl_agent.agent import RLAgent  # 现在这是一个 nn.Module


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
        self.diffusion_model = DiffusionModel(config).to(self.device)  # 传递整个 config

        # --- 加载检查点 (如果指定) ---
        self.checkpoint_loaded = False  # 标记是否成功加载了检查点
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
                    self.checkpoint_loaded = True
                    print("--- DiT 模型权重加载成功 ---")
                except Exception as e:
                    print(f"[警告] 加载检查点失败: {e}")
                    print("将使用随机初始化的权重。")
            else:
                print(f"[警告] 指定的检查点文件不存在: {checkpoint_path}")
                print("将使用随机初始化的权重。")
        else:
            print("--- 未指定加载检查点, DiT 模型使用随机初始化权重 ---")

        self.evaluator = MechanismEvaluator(config)  # 传递整个 config
        self.rl_agent = RLAgent(config).to(self.device)  # 传递整个 config

        # --- 4. 设置优化器 (现在有两个!) ---
        self.dit_optimizer = optim.AdamW(
            self.diffusion_model.parameters(),
            lr=float(config['training'].get('learning_rate', 0.0001))  # 确保是 float
        )
        self.rl_optimizer = optim.AdamW(
            self.rl_agent.parameters(),
            lr=float(config['training'].get('rl_learning_rate', 0.00001))  # 确保是 float
        )

        # --- 5. 加载实验开关 ---
        self.enable_rl_guidance = config['training'].get('enable_rl_guidance', True)
        self.enable_augmentation = config['training'].get('enable_augmentation', True)
        self.acceptance_threshold = config['generation'].get('acceptance_threshold', -float('inf'))

        # --- 获取目标生成标签 ---
        self.target_label_index = config['generation'].get('target_label_index', 0)

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
        if x_tensor.dim() == 3:  # (C, H, W) -> (1, C, H, W)
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
            return result.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        return result

    def _unnormalize(self, x_tensor_norm):
        """(新!) 将 (B, 4, H, W) 或 (C, H, W) 数据从 [-1, 1] 恢复."""
        if x_tensor_norm.dim() == 3:  # (C, H, W) -> (1, C, H, W)
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
            return result.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        return result

    # --- 加载检查点 ---
    def _load_checkpoint(self):
        load_path_rel = self.config['training'].get('load_checkpoint_path')
        if load_path_rel and load_path_rel.lower() != 'null':
            checkpoint_path = os.path.join(self.project_root, load_path_rel)
            if os.path.exists(checkpoint_path):
                try:
                    print(f"--- 正在从检查点加载 DiT 模型权重: {checkpoint_path} ---")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    # 尝试加载模型状态, 忽略不匹配的键 (例如 label_embed 可能不存在于旧检查点)
                    missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    if missing_keys:
                        print(f"[警告] 加载检查点时缺少键: {missing_keys}")
                    if unexpected_keys:
                        print(f"[警告] 加载检查点时有多余键: {unexpected_keys}")
                    print("--- DiT 模型权重加载完成 ---")
                except Exception as e:
                    print(f"[警告] 加载检查点失败: {e}. 将使用随机初始化的权重。")
            else:
                print(f"[警告] 检查点文件不存在: {checkpoint_path}. 将使用随机初始化的权重。")
        else:
            print("--- 未指定加载检查点, DiT 模型使用随机初始化权重 ---")

    def load_replay_buffer(self):  # (已更新, 读取标签并归一化)
        print("正在加载 Replay Buffer (来自 augmented_dataset)...")
        manifest_path = os.path.join(self.project_root, self.config['data']['augmented_manifest_path'])  # 使用绝对路径
        data_dir = os.path.dirname(manifest_path)
        if not os.path.exists(manifest_path):
            print("信息: augmented_manifest.json 不存在, Replay Buffer 为空。")
            return []
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"错误: 读取 manifest {manifest_path} 失败: {e}")
            return []
        experiences = []
        label_map = {"bennett": 0}  # 保持简单
        for entry in manifest:
            npz_path = os.path.join(data_dir, entry['data_path'])
            try:
                # --- (新!) 检查标签 ---
                label_str = entry.get('metadata', {}).get('label')
                if label_str is None:
                    print(f"警告: 条目 {entry['id']} 缺少标签, 跳过。")
                    continue
                label_idx = label_map.get(label_str, -1)
                if label_idx == -1:
                    print(f"警告: 条目 {entry['id']} 标签 '{label_str}' 未知, 跳过。")
                    continue

                with np.load(npz_path) as data:
                    tensor_unnorm_numpy = data['mechanism_tensor']
                    tensor_unnorm_torch = torch.from_numpy(np.transpose(tensor_unnorm_numpy, (2, 0, 1))).float()
                    tensor_norm = self._normalize(tensor_unnorm_torch)
                    score = entry.get('metadata', {}).get('score', 0.0)  # 提供默认值
                    # 存储 (归一化张量, 分数, 标签索引)
                    experiences.append((tensor_norm, score, torch.tensor(label_idx, dtype=torch.long)))
            except Exception as e:
                print(f"警告: 无法加载 Replay Buffer 条目 {npz_path}. 原因: {e}")
        print(f"Replay Buffer 加载完成. 共 {len(experiences)} 个历史经验。")
        return experiences

    def _generate_and_augment(self, cycle_num):  # (已更新, 传递 target_label)
        print("\n--- 步骤 1 & 2: 生成, 评估, 扩充 ---")
        self.diffusion_model.eval()
        self.rl_agent.eval()
        guidance_fn = None
        if self.enable_rl_guidance:
            print("RL 引导已启用。")
            guidance_fn = self.rl_agent.get_guidance_fn(self.guidance_scale)
        else:
            print("RL 引导已关闭。将使用纯 DiT 采样。")

        # --- (新!) 准备目标标签 ---
        num_to_gen = self.config['generation']['num_to_generate']
        # 创建一个包含 num_to_gen 个目标标签索引的张量
        target_labels = torch.full((num_to_gen,), self.target_label_index, dtype=torch.long, device=self.device)

        # --- (新!) 调用 sample 时传入 target_labels ---
        new_mech_tensors_unnorm_numpy = self.diffusion_model.sample(
            num_samples=num_to_gen,
            y=target_labels,  # <-- 传入目标标签
            guidance_fn=guidance_fn,
            guidance_scale=self.guidance_scale
        )

        print(f"评估 {len(new_mech_tensors_unnorm_numpy)} 个新生成的机构...")
        new_experiences_for_rl = []
        good_mechanisms_to_save = []
        current_target_label_str = "bennett"  # TODO: 从索引反查

        for tensor_unnorm_numpy in new_mech_tensors_unnorm_numpy:
            score = self.evaluator.evaluate(tensor_unnorm_numpy)
            tensor_torch_unnorm = torch.from_numpy(np.transpose(tensor_unnorm_numpy, (2, 0, 1))).float()
            tensor_torch_norm = self._normalize(tensor_torch_unnorm)

            # --- (新!) 经验中包含标签索引 ---
            new_experiences_for_rl.append(
                (tensor_torch_norm.to(self.device), score, torch.tensor(self.target_label_index, dtype=torch.long)))

            if self.enable_augmentation and score >= self.acceptance_threshold:
                # --- (新!) 保存时也包含标签 ---
                new_entry = {
                    "tensor": tensor_unnorm_numpy,
                    "metadata": {
                        "source": "generated", "generation_cycle": cycle_num + 1,
                        "score": score, "label": current_target_label_str  # <-- 保存标签字符串
                    }}
                good_mechanisms_to_save.append(new_entry)

        print(f"评估完成. {len(good_mechanisms_to_save)} / {len(new_mech_tensors_unnorm_numpy)} 个机构满足保存要求...")
        if self.enable_augmentation:
            if good_mechanisms_to_save:
                print(f"数据增强已启用。正在保存 {len(good_mechanisms_to_save)} 个机构...")
                dataloader.add_mechanisms_to_dataset(good_mechanisms_to_save,
                                                     os.path.join(self.project_root, self.config['data'][
                                                         'augmented_manifest_path']))  # 使用绝对路径
            else:
                print("数据增强已启用, 但没有合格的机构可保存。")
        else:
            print("数据增强已关闭。跳过保存。")
        return new_experiences_for_rl

    def _train_rl_agent(self, new_experiences):  # (已更新, 处理带标签的经验)
        if not self.enable_rl_guidance:
            print("\n--- RL 引导已关闭, 跳过 RL 智能体训练 ---")
            return
        print("\n--- 步骤 3a: 训练 RL 智能体 ---")
        buffer_limit = self.config['training'].get('replay_buffer_limit', 50000)
        self.replay_buffer.extend(new_experiences)  # new_experiences 是 (tensor_norm, score, label_idx)
        if len(self.replay_buffer) > buffer_limit:
            self.replay_buffer = self.replay_buffer[-buffer_limit:]
        print(f"Replay Buffer 中共有 {len(self.replay_buffer)} 个经验。")
        if not self.replay_buffer:
            print("Replay Buffer 为空, 跳过 RL 训练。")
            return

        # --- (新!) update_policy 可能需要标签, 但我们当前实现不需要 ---
        # 如果 RLAgent 的 forward 或 loss 需要标签, 需要修改 update_policy
        self.rl_agent.update_policy(
            self.replay_buffer,  # 包含 (归一化张量, 分数, 标签索引)
            self.diffusion_model,
            self.rl_optimizer,
            self.device
            # 如果需要, 可以传递标签信息给 update_policy
        )

    def _train_dit_model(self, cycle_num):  # (已更新, 处理标签)
        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])
        augmented_manifest_path = None
        if self.enable_augmentation:
            print("数据增强已启用。DiT 将在 [初始 + 增强] 数据集上训练。")
            augmented_manifest_path = os.path.join(self.project_root, self.config['data']['augmented_manifest_path'])
        else:
            print("数据增强已关闭。DiT 将仅在 [初始] 数据集上训练。")

        current_dataloader = dataloader.get_dataloader(
            self.config,  # <-- 传递 config
            initial_manifest_path, augmented_manifest_path,
            self.config['training']['batch_size'], shuffle=True)
        if current_dataloader is None or len(current_dataloader.dataset) == 0:
            print("错误：数据加载失败或数据集为空, 跳过 DiT 训练。")
            return

        print(f"\n--- 步骤 3b: 训练 DiT (模仿者) ---")
        print(f"将使用 {len(current_dataloader.dataset)} 个机构进行训练。")
        self.diffusion_model.train()
        num_epochs = self.config['training']['epochs_per_cycle']
        for epoch in range(num_epochs):
            total_loss = 0.0
            # --- (核心修改!) Dataloader 返回 (x_start, y_label) ---
            for x_start, y_labels in current_dataloader:
                x_start = x_start.to(self.device)  # 未归一化
                y_labels = y_labels.to(self.device)  # 标签索引
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                # --- q_sample 接收未归一化 x_start, 返回归一化 x_t_norm ---
                # (我们当前的 q_sample 实现不依赖 y_labels)
                x_start_norm = self.diffusion_model._normalize(x_start)  # 手动归一化
                noise = torch.randn_like(x_start_norm)  # 噪声与归一化同形状
                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)

                # --- (核心修改!) forward 接收归一化 x_t, t, 和 y_labels ---
                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)

                loss = F.mse_loss(predicted_noise_norm, noise)
                loss.backward()
                self.dit_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  DiT Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    def _warmup_dit(self, num_epochs):
        """
        (已修正!) DiT 预热训练: 在主循环之前, 先在初始数据集上预训练 DiT 模型.
        """
        print("\n" + "="*60)
        print(f"--- DiT 预热阶段: 在初始数据集上训练 {num_epochs} 轮 ---")
        print("="*60)

        # 加载初始数据集 (不包含增强数据集)
        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])  # 使用绝对路径
        warmup_dataloader = dataloader.get_dataloader(
            self.config,
            initial_manifest_path,
            None,  # 预热阶段不使用增强数据集
            self.config['training']['batch_size'],
            shuffle=True
        )

        if warmup_dataloader is None or len(warmup_dataloader.dataset) == 0:
            print("[警告] 初始数据集为空, 跳过预热阶段。")
            return

        print(f"初始数据集大小: {len(warmup_dataloader.dataset)} 个机构")

        # 设置为训练模式
        self.diffusion_model.train()

        # 训练循环 (与 _train_dit_model 逻辑保持一致)
        for epoch in range(num_epochs):
            total_loss = 0.0

            # Dataloader 返回 (x_start_unnorm, y_labels)
            for x_start, y_labels in warmup_dataloader:
                x_start = x_start.to(self.device)  # 未归一化
                y_labels = y_labels.to(self.device)  # 标签索引
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                # --- 核心修正: 与 _train_dit_model 保持一致 ---
                # 1. 归一化 x_start
                x_start_norm = self._normalize(x_start)

                # 2. 创建标准高斯噪声 (与归一化数据同形状)
                noise = torch.randn_like(x_start_norm)

                # 3. q_sample 接收 *归一化* 输入, 返回归一化 x_t
                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)

                # 4. forward 接收归一化 x_t, t, 和 y_labels, 预测归一化 noise
                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)

                # 5. 损失在归一化空间计算
                loss = F.mse_loss(predicted_noise_norm, noise)  # 比较归一化预测 vs 标准高斯噪声
                # --- 修正结束 ---

                loss.backward()
                self.dit_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(warmup_dataloader)

            # 每10轮或最后一轮打印一次
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  预热 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

        print("\n--- DiT 预热完成! 模型已学习初始数据集的基本结构 ---")
        print("="*60 + "\n")

    def run(self):
        """
        执行完整的三步循环：生成 -> 训练RL -> 训练DiT
        """
        # --- DiT 预热阶段 ---
        # 如果没有加载预训练模型, 则先在初始数据集上预热 DiT
        warmup_epochs = self.config['training'].get('dit_warmup_epochs', 0)

        if warmup_epochs > 0 and not self.checkpoint_loaded:
            print("\n[检测到] DiT 使用随机初始化权重, 将进行预热训练...")
            self._warmup_dit(warmup_epochs)
        elif warmup_epochs > 0 and self.checkpoint_loaded:
            print("\n[跳过预热] DiT 已从检查点加载, 无需预热。")
        else:
            print("\n[跳过预热] 配置文件中 dit_warmup_epochs=0, 不进行预热。")

        # --- 主训练循环 ---
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
            self.save_checkpoint(f"cycle_{cycle + 1}")

        print("\n===== 所有训练循环完成! =====")

        # --- (新!) 在训练结束后保存最终模型 ---
        self.save_checkpoint("final")  # 调用保存函数

    # --- 添加保存检查点的函数 ---
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
