# src/pipeline.py

import os
import json
import logging
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ( ... 导入 ... )
from .utils import dataloader
from .evaluator.evaluator import MechanismEvaluator
from .diffusion_model.model import DiffusionModel
from .rl_agent.agent import RLAgent


class TrainingPipeline:
    def __init__(self, config, project_root):
        # --- 获取 Logger ---
        self.logger = logging.getLogger()
        self.config = config
        self.project_root = project_root
        self.device = self._setup_device()

        # --- 2. 加载数据归一化参数 (4 通道) ---
        try:
            norm_values_dict = config['data']['normalization_values']
            # norm_vec 是 4 通道 (mixed, a, alpha, offset/d)
            norm_vec = torch.tensor([
                norm_values_dict['mixed'],
                norm_values_dict['a'],
                norm_values_dict['alpha'],
                norm_values_dict['d']
            ], dtype=torch.float32, device=self.device)
            self.norm_vec = norm_vec.view(1, 4, 1, 1)  # (1, 4, 1, 1)
        except KeyError:
            raise ValueError("[致命错误] 配置文件 data.normalization_values 块缺少 'mixed', 'a', 'alpha' 或 'd'。")

        self.guidance_scale = config['generation'].get('rl_guidance_scale', 1.0)

        # --- 3. 初始化所有模块 ---
        self.logger.info("--- 正在初始化所有模块... ---")
        self.diffusion_model = DiffusionModel(config).to(self.device)
        self.evaluator = MechanismEvaluator(config)
        self.rl_agent = RLAgent(config).to(self.device)

        # --- 4. 设置优化器 ---
        self.dit_optimizer = optim.AdamW(
            self.diffusion_model.parameters(),
            lr=float(config['training'].get('learning_rate', 0.0001))
        )
        self.rl_optimizer = optim.AdamW(
            self.rl_agent.parameters(),
            lr=float(config['training'].get('rl_learning_rate', 0.00001))
        )

        # --- 5. 初始化状态属性并加载检查点 ---
        self.checkpoint_loaded = False
        self.start_cycle = 0
        self.replay_buffer = []
        load_path = config['training'].get('load_checkpoint_path')
        if load_path and load_path.lower() != 'null':
            checkpoint_path = os.path.join(project_root, load_path)
            if os.path.exists(checkpoint_path):
                try:
                    self.logger.info(f"--- 正在从【完整】检查点加载状态 ---")
                    self.logger.info(f"路径: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    # 5a. 加载 DiT 模型
                    self.diffusion_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    self.logger.info("  > DiT 模型权重... OK")
                    # 5b. 加载 RL Agent
                    if 'rl_agent_state_dict' in checkpoint:
                        self.rl_agent.load_state_dict(checkpoint['rl_agent_state_dict'], strict=True)
                        self.logger.info("  > RL Agent 权重... OK")
                    else:
                        self.logger.warning(
                            "  > 警告: 检查点中未找到 'rl_agent_state_dict'. RL Agent 将使用随机初始化。")
                    # 5c. 加载优化器状态
                    if 'dit_optimizer_state_dict' in checkpoint:
                        self.dit_optimizer.load_state_dict(checkpoint['dit_optimizer_state_dict'])
                        self.logger.info("  > DiT 优化器状态... OK")
                    if 'rl_optimizer_state_dict' in checkpoint:
                        self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
                        self.logger.info("  > RL 优化器状态... OK")
                    # 5d. 加载训练进度和 Replay Buffer
                    if 'cycle' in checkpoint:
                        self.start_cycle = checkpoint['cycle'] + 1
                        self.logger.info(
                            f"  > 训练进度... OK (将从 cycle {self.start_cycle} / {self.config['training']['num_cycles']} 处恢复)")
                    if 'replay_buffer' in checkpoint:
                        loaded_replay_buffer = checkpoint['replay_buffer']
                        self.replay_buffer = []  # 重置
                        for (tensor_cpu, score, label_cpu) in loaded_replay_buffer:
                            self.replay_buffer.append(
                                (tensor_cpu.to(self.device), score, label_cpu.to(self.device))
                            )
                        self.logger.info(f"  > Replay Buffer... OK (已从检查点加载 {len(self.replay_buffer)} 个经验)")
                    self.checkpoint_loaded = True
                    self.logger.info("--- 完整检查点加载成功 ---")
                except Exception as e:
                    self.logger.warning(f"[警告] 加载检查点失败: {e}")
                    self.logger.warning("将使用随机初始化的权重。")
            else:
                self.logger.warning(f"[警告] 指定的检查点文件不存在: {checkpoint_path}")
                self.logger.warning("将使用随机初始化的权重。")
        if not self.checkpoint_loaded:
            self.logger.info("--- 未指定或加载检查点失败, 所有模块使用随机初始化权重 ---")
            self.replay_buffer = self.load_replay_buffer()
        # --- 6. 加载实验开关 ---
        self.enable_rl_guidance = config['training'].get('enable_rl_guidance', True)
        self.enable_augmentation = config['training'].get('enable_augmentation', True)
        self.acceptance_threshold = config['generation'].get('acceptance_threshold', -float('inf'))
        self.rl_warmup_epochs = config['training'].get('rl_warmup_epochs', 0)
        self.target_label_index = config['generation'].get('target_label_index', 0)
        self.logger.info("--- 训练流程已准备就绪 ---")
        self.logger.info(f"  RL 引导: {self.enable_rl_guidance}")
        self.logger.info(f"  数据增强: {self.enable_augmentation}")
        if self.enable_augmentation:
            self.logger.info(f"  增强阈值 (分数 >=): {self.acceptance_threshold}")

    def _setup_device(self):
        #
        if self.config['training'].get('device', 'cpu') == 'cuda' and torch.cuda.is_available():
            self.logger.info("训练将在 NVIDIA CUDA GPU 上运行。")
            return torch.device("cuda")
        else:
            self.logger.info("训练将在 CPU 上运行。")
            return torch.device("cpu")

    # --- 归一化/反归一化 辅助函数 (4 通道, 统一) ---
    def _normalize(self, x_tensor):
        """ 将 (B, 4, H, W) 或 (C, 4, H, W) 数据按通道归一化到 [-1, 1]."""
        original_dim = x_tensor.dim()
        if original_dim == 3:
            x_tensor = x_tensor.unsqueeze(0);
            was_3d = True
        else:
            was_3d = False

        if x_tensor.shape[1] != 4: raise ValueError(f"通道数应为4, 收到 {x_tensor.shape[1]}")

        x_tensor = x_tensor.to(self.device)

        # self.norm_vec (1, 4, 1, 1) 已经在 self.device 上
        result = (x_tensor / (self.norm_vec + 1e-8)) * 2.0 - 1.0

        if was_3d: return result.squeeze(0)
        return result

    def _unnormalize(self, x_tensor_norm):
        """ 将 (B, 4, H, W) 或 (C, 4, H, W) 数据从 [-1, 1] 按通道恢复."""
        original_dim = x_tensor_norm.dim()
        if original_dim == 3:
            x_tensor_norm = x_tensor_norm.unsqueeze(0);
            was_3d = True
        else:
            was_3d = False

        if x_tensor_norm.shape[1] != 4: raise ValueError(f"通道数应为4, 收到 {x_tensor_norm.shape[1]}")

        x_tensor_norm = x_tensor_norm.to(self.device)

        # self.norm_vec (1, 4, 1, 1) 已经在 self.device 上
        result = ((x_tensor_norm + 1.0) / 2.0) * self.norm_vec

        if was_3d: return result.squeeze(0)
        return result

    def _load_checkpoint(self):
        # (此函数似乎未被调用, 但保持原样)
        load_path_rel = self.config['training'].get('load_checkpoint_path')
        if load_path_rel and load_path_rel.lower() != 'null':
            checkpoint_path = os.path.join(self.project_root, load_path_rel)
            if os.path.exists(checkpoint_path):
                try:
                    self.logger.info(f"--- 正在从检查点加载 DiT 模型权重: {checkpoint_path} ---")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    if missing_keys: self.logger.warning(f"[警告] 加载检查点时缺少键: {missing_keys}")
                    if unexpected_keys: self.logger.warning(f"[警告] 加载检查点时有多余键: {unexpected_keys}")
                    self.logger.info("--- DiT 模型权重加载完成 ---")
                except Exception as e:
                    self.logger.warning(f"[警告] 加载检查点失败: {e}. 将使用随机初始化的权重。")
            else:
                self.logger.warning(f"[警告] 检查点文件不存在: {checkpoint_path}. 将使用随机初始化的权重。")
        else:
            self.logger.info("--- 未指定加载检查点, DiT 模型使用随机初始化权重 ---")

    def load_replay_buffer(self):
        """
        从 augmented_dataset 加载经验 (6列稀疏 + 1D joint_types -> 4通道稠密).
        """
        self.logger.info("正在加载 Replay Buffer (来自 augmented_dataset)...")
        manifest_path = os.path.join(self.project_root, self.config['data']['augmented_manifest_path'])
        data_dir = os.path.dirname(manifest_path)

        if not os.path.exists(manifest_path):
            self.logger.info("信息: augmented_manifest.json 不存在, Replay Buffer 为空。")
            return []

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            self.logger.error(f"错误: 读取 manifest {manifest_path} 失败: {e}")
            return []

        experiences = []
        label_map = {"bennett": 0}

        for entry in manifest:
            npz_path = os.path.join(data_dir, entry['data_path'])
            try:
                # --- 检查标签 ---
                label_str = entry.get('metadata', {}).get('label')
                if label_str is None: continue
                label_idx = label_map.get(label_str, -1)
                if label_idx == -1: continue

                with np.load(npz_path) as data:
                    if 'edge_list_array' not in data or 'joint_types_array' not in data:
                        self.logger.warning(f"警告: {npz_path} 中缺少 'edge_list_array' 或 'joint_types_array', 跳过。")
                        continue

                    edge_list_array = data['edge_list_array']  # (N, 6)
                    joint_types_array = data['joint_types_array']  # (M,)

                    # 像 Dataloader 一样动态构建稠密张量
                    max_nodes = self.config['data']['max_nodes']
                    num_features = self.config['diffusion_model']['in_channels']  # 4
                    feature_tensor = np.zeros((max_nodes, max_nodes, num_features), dtype=np.float32)

                    # 1. 填充对角线 (Ch 0: 关节类型)
                    for i, joint_type in enumerate(joint_types_array):
                        if i >= max_nodes: break
                        feature_tensor[i, i, 0] = joint_type

                    # 2. 填充非对角线 (边)
                    for edge_data in edge_list_array:
                        # 解包 6 个值
                        i, k, a, alpha, d_i, d_k = edge_data
                        i, k = int(i), int(k)
                        if i >= max_nodes or k >= max_nodes: continue

                        # 填充 4 通道
                        # [mixed, a, alpha, offset]
                        feature_tensor[i, k] = [1.0, a, alpha, d_i]
                        feature_tensor[k, i] = [1.0, a, alpha, d_k]
                        # 确保 a 和 alpha 是对称的
                        feature_tensor[k, i, 1] = a
                        feature_tensor[k, i, 2] = alpha

                    # (H, W, C=4) -> (C=4, H, W)
                    tensor_unnorm_torch = torch.from_numpy(np.transpose(feature_tensor, (2, 0, 1))).float()

                    tensor_norm = self._normalize(tensor_unnorm_torch)  # 使用 pipeline 的 4-ch normalizer
                    score = entry.get('metadata', {}).get('score', 0.0)

                    experiences.append((tensor_norm, score, torch.tensor(label_idx, dtype=torch.long)))

            except Exception as e:
                self.logger.warning(f"警告: 无法加载 Replay Buffer 条目 {npz_path}. 原因: {e}")

        self.logger.info(f"Replay Buffer 加载完成. 共 {len(experiences)} 个历史经验。")
        return experiences

    def _generate_and_augment(self, cycle_num, total_cycles):
        """
        步骤 1 & 2: 生成 (带引导), 评估, 并扩充数据集.
        返回: 新的经验 [(pure_x0_norm, score, label_idx), ...], 用于训练 RL Agent.
        """
        self.logger.info("--- 步骤 1 & 2: 生成, 评估, 扩充 ---")
        self.diffusion_model.eval()
        self.rl_agent.eval()

        guidance_fn = None
        if self.enable_rl_guidance:
            self.logger.info("RL 引导已启用。")
            guidance_fn = self.rl_agent.get_guidance_fn(self.guidance_scale)
        else:
            self.logger.info("RL 引导已关闭。将使用纯 DiT 采样。")

        num_to_gen = self.config['generation']['num_to_generate']
        target_labels = torch.full((num_to_gen,), self.target_label_index, dtype=torch.long, device=self.device)

        label_map = {0: "bennett", 1: "planar_four_bar"}
        current_target_label_str = label_map.get(self.target_label_index, "bennett")
        self.logger.info(f"正在生成 {num_to_gen} 个目标标签为 '{current_target_label_str}' 的机构...")

        # --- 接收两个输出 ---
        # 1. new_mech_tensors_unnorm_numpy: list of (H, W, 4) numpy 数组 (已后处理)
        # 2. batch_x0_norm_pure: (B, 4, H, W) torch 张量 (未后处理, 已归一化)
        new_mech_tensors_unnorm_numpy, batch_x0_norm_pure = self.diffusion_model.sample(
            num_samples=num_to_gen,
            y=target_labels,
            guidance_fn=guidance_fn,
            guidance_scale=self.guidance_scale
        )  #

        self.logger.info(f"评估 {len(new_mech_tensors_unnorm_numpy)} 个新生成的机构...")
        new_experiences_for_rl = []
        good_mechanisms_to_save = []
        num_satisfying_score = 0
        current_target_label_str = "bennett"  # TODO: 从索引反查

        # --- 使用 enumerate 获取索引 ---
        for i, tensor_unnorm_numpy in enumerate(new_mech_tensors_unnorm_numpy):  # (H, W, 4)

            # 1. 评估器使用 "后处理" 的 numpy 张量
            score = self.evaluator.evaluate(
                tensor_unnorm_numpy,
                target_label=current_target_label_str,
                current_cycle=cycle_num,
                total_cycles=total_cycles
            )  #

            # 2. 从 batch 中获取 "纯粹" 的 x_0_norm
            #    我们不再需要 _normalize()
            tensor_torch_norm_pure = batch_x0_norm_pure[i].to(self.device)  # (4, H, W)

            # 3. 将 "纯粹" 的 x_0_norm 存入缓冲区
            new_experiences_for_rl.append(
                (tensor_torch_norm_pure, score, torch.tensor(self.target_label_index, dtype=torch.long)))  #

            # 4. 检查分数
            if score >= self.acceptance_threshold:
                num_satisfying_score += 1

                if self.enable_augmentation:
                    # 我们仍然保存 "后处理" 的 numpy 张量 (H, W, 4)
                    new_entry = {
                        "tensor": tensor_unnorm_numpy,
                        "metadata": {
                            "source": "generated", "generation_cycle": cycle_num + 1,
                            "score": score, "label": current_target_label_str
                        }}
                    good_mechanisms_to_save.append(new_entry)

        if new_experiences_for_rl:
            all_scores = [exp[1] for exp in new_experiences_for_rl]
            avg_score = sum(all_scores) / len(all_scores)
            min_score = min(all_scores);
            max_score = max(all_scores)
            self.logger.info(f"生成机构平均得分: {avg_score:.4f} (Min: {min_score:.4f}, Max: {max_score:.4f})")
        else:
            avg_score = None

        self.logger.info(
            f"评估完成. {num_satisfying_score} / {len(new_mech_tensors_unnorm_numpy)} 个机构满足分数要求...")

        if self.enable_augmentation:
            if good_mechanisms_to_save:
                self.logger.info(f"数据增强已启用。正在保存 {len(good_mechanisms_to_save)} 个机构...")
                dataloader.add_mechanisms_to_dataset(good_mechanisms_to_save,
                                                     os.path.join(self.project_root, self.config['data'][
                                                         'augmented_manifest_path']))
            else:
                self.logger.info("数据增强已启用, 但没有合格的机构可保存。")
        else:
            self.logger.info("数据增强已关闭。跳过保存。")

        return new_experiences_for_rl

    def _train_rl_agent(self, new_experiences):
        # 在归一化空间操作, RLAgent 已被修改为 4 通道
        if not self.enable_rl_guidance:
            self.logger.info("--- RL 引导已关闭, 跳过 RL 智能体训练 ---")
            return
        self.logger.info("--- 步骤 3a: 训练 RL 智能体 ---")
        buffer_limit = self.config['training'].get('replay_buffer_limit', 50000)
        self.replay_buffer.extend(new_experiences)
        if len(self.replay_buffer) > buffer_limit:
            self.replay_buffer = self.replay_buffer[-buffer_limit:]
        self.logger.info(f"Replay Buffer 中共有 {len(self.replay_buffer)} 个经验。")
        if not self.replay_buffer:
            self.logger.info("Replay Buffer 为空, 跳过 RL 训练。")
            return
        avg_rl_loss = self.rl_agent.update_policy(
            self.replay_buffer,  #
            self.diffusion_model,
            self.rl_optimizer,
            self.device
        )
        if avg_rl_loss is not None:
            self.logger.info(f"--- RL Agent 训练完成. 平均 Loss: {avg_rl_loss:.6f} ---")

    def _train_dit_model(self, cycle_num):
        # Dataloader 和 Model 都已修改为 4 通道
        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])
        augmented_manifest_path = None
        if self.enable_augmentation:
            self.logger.info("数据增强已启用。DiT 将在 [初始 + 增强] 数据集上训练。")
            augmented_manifest_path = os.path.join(self.project_root, self.config['data']['augmented_manifest_path'])
        else:
            self.logger.info("数据增强已关闭。DiT 将仅在 [初始] 数据集上训练。")

        current_dataloader = dataloader.get_dataloader(
            self.config,
            initial_manifest_path, augmented_manifest_path,
            self.config['training']['batch_size'], shuffle=True)
        if current_dataloader is None or len(current_dataloader.dataset) == 0:
            self.logger.error("错误：数据加载失败或数据集为空, 跳过 DiT 训练。")
            return

        self.logger.info(f"--- 步骤 3b: 训练 DiT (模仿者) ---")
        self.logger.info(f"将使用 {len(current_dataloader.dataset)} 个机构进行训练。")
        self.diffusion_model.train()
        num_epochs = self.config['training']['epochs_per_cycle']
        for epoch in range(num_epochs):
            total_loss = 0.0
            # Dataloader 返回 (x_start(B, 4, H, W), y_label)
            for x_start, y_labels in current_dataloader:
                x_start = x_start.to(self.device)
                y_labels = y_labels.to(self.device)
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                x_start_norm = self.diffusion_model._normalize(x_start)  # 使用 model 的 4-ch normalizer
                noise = torch.randn_like(x_start_norm)  # (B, 4, H, W)

                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)  #

                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)  #

                loss = F.mse_loss(predicted_noise_norm, noise)
                loss.backward()
                self.dit_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"  DiT Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    def _warmup_dit(self, num_epochs):
        # Dataloader 和 _normalize 都已修改
        self.logger.info(f"--- DiT 预热阶段: 在初始数据集上训练 {num_epochs} 轮 ---")
        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])
        warmup_dataloader = dataloader.get_dataloader(
            self.config, initial_manifest_path, None,
            self.config['training']['batch_size'], shuffle=True)
        if warmup_dataloader is None or len(warmup_dataloader.dataset) == 0:
            self.logger.warning("[警告] 初始数据集为空, 跳过预热阶段。")
            return
        self.logger.info(f"初始数据集大小: {len(warmup_dataloader.dataset)} 个机构")
        self.diffusion_model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            # Dataloader 返回 (x_start(B, 4, H, W), y_labels)
            for x_start, y_labels in warmup_dataloader:
                x_start = x_start.to(self.device)
                y_labels = y_labels.to(self.device)
                batch_size = x_start.size(0)
                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                x_start_norm = self._normalize(x_start)  # 使用 pipeline 的 4-ch normalizer
                noise = torch.randn_like(x_start_norm)  # (B, 4, H, W)

                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)  #

                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)  #

                loss = F.mse_loss(predicted_noise_norm, noise)
                loss.backward()
                self.dit_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(warmup_dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"  预热 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")
        self.logger.info("--- DiT 预热完成! 模型已学习初始数据集的基本结构 ---")

    def _warmup_rl_agent(self, num_epochs):
        # Dataloader, _normalize, evaluator 都已修改
        self.logger.info(f"--- RL 智能体预热阶段: 在初始数据集上训练 {num_epochs} 轮 ---")
        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])

        initial_dataset = dataloader.MechanismDataset(
            self.config, initial_manifest_path, None
        )
        if len(initial_dataset) == 0:
            self.logger.warning("[警告] 初始数据集为空, 跳过 RL 预热。")
            return
        self.logger.info(f"正在从 {len(initial_dataset)} 个初始样本中预填充 Replay Buffer...")
        temp_loader = torch.utils.data.DataLoader(initial_dataset,
                                                  batch_size=self.config['training']['batch_size'],
                                                  shuffle=False)
        experiences_to_add = []
        for x_start_batch, y_labels_batch in temp_loader:
            for i in range(x_start_batch.size(0)):
                x_start_unnorm = x_start_batch[i]  # (4, H, W)
                y_label = y_labels_batch[i]

                x_start_norm = self._normalize(x_start_unnorm)  # (4, H, W)

                # (4, H, W) -> (H, W, 4)
                x_start_unnorm_numpy = x_start_unnorm.permute(1, 2, 0).cpu().numpy()

                score = self.evaluator.evaluate(x_start_unnorm_numpy)  #

                experiences_to_add.append(
                    (x_start_norm.to(self.device), score, y_label.to(self.device))
                )
        self.replay_buffer.extend(experiences_to_add)
        self.logger.info(f"Replay Buffer 预填充完成. 当前总经验: {len(self.replay_buffer)}")
        self.logger.info(f"开始在 {len(self.replay_buffer)} 个样本上预训练 RL Agent...")
        total_loss = 0
        num_updates = 0
        for epoch in range(num_epochs):
            avg_rl_loss = self.rl_agent.update_policy(
                self.replay_buffer,
                self.diffusion_model,
                self.rl_optimizer,
                self.device
            )
            if avg_rl_loss is not None:
                total_loss += avg_rl_loss
                num_updates += 1
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1 or num_epochs <= 10:
                self.logger.info(
                    f"  RL 预热 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | Avg MSE Loss: {avg_rl_loss:.6f}")
        if num_updates > 0:
            self.logger.info(f"--- RL 智能体预热完成! 平均 Loss: {total_loss / num_updates:.6f} ---")
        else:
            self.logger.info("--- RL 智能体预热完成 (没有更新). ---")

    def run(self):
        #
        warmup_epochs = self.config['training'].get('dit_warmup_epochs', 0)
        if warmup_epochs > 0 and not self.checkpoint_loaded:
            self.logger.info("[检测到] DiT 使用随机初始化权重, 将进行预热训练...")
            self._warmup_dit(warmup_epochs)
        elif self.checkpoint_loaded:
            self.logger.info(f"[跳过预热] DiT 已从检查点加载 (将从 cycle {self.start_cycle} 开始)。")
        else:
            self.logger.info("[跳过预热] 配置文件中 dit_warmup_epochs=0, 不进行预热。")

        if self.rl_warmup_epochs > 0 and self.enable_rl_guidance and not self.replay_buffer:
            self.logger.info("[检测到] RL 引导已启用, 但 Replay Buffer 为空. 将进行 RL 预热...")
            self._warmup_rl_agent(self.rl_warmup_epochs)
        elif self.checkpoint_loaded:
            self.logger.info(f"[跳过 RL 预热] Replay Buffer 已从检查点加载 ({len(self.replay_buffer)} 个经验)。")
        elif self.rl_warmup_epochs > 0 and self.replay_buffer:
            self.logger.info(f"[跳过 RL 预热] Replay Buffer 已包含 {len(self.replay_buffer)} 个经验, 无需预热。")
        elif self.rl_warmup_epochs > 0 and not self.enable_rl_guidance:
            self.logger.info("[跳过 RL 预热] RL 引导已关闭。")
        else:
            self.logger.info("[跳过 RL 预热] 配置文件中 rl_warmup_epochs=0, 不进行预热。")

        num_cycles = self.config['training']['num_cycles']
        self.logger.info(f"--- 开始总共 {num_cycles} 轮的训练循环 ---")
        for cycle in range(self.start_cycle, num_cycles):
            self.logger.info(f"===== [ 完整循环 {cycle + 1}/{num_cycles} ] =====")
            self.logger.info("\n")
            start_gen_eval = time.time()
            new_experiences = self._generate_and_augment(cycle, num_cycles)
            self.logger.info(f"  生成与评估耗时: {time.time() - start_gen_eval:.2f} 秒")
            self.logger.info("\n")
            start_rl_train = time.time()
            self._train_rl_agent(new_experiences)
            self.logger.info(f"  RL 训练耗时: {time.time() - start_rl_train:.2f} 秒")
            self.logger.info("\n")
            start_dit_train = time.time()
            self._train_dit_model(cycle)
            self.logger.info(f"  DiT 训练耗时: {time.time() - start_dit_train:.2f} 秒")
            self.logger.info("\n")
            self.save_checkpoint(cycle_num=cycle)
        self.logger.info("===== 所有训练循环完成! =====")
        self.save_checkpoint(cycle_num=num_cycles)

    def save_checkpoint(self, cycle_num=0):
        #
        save_path_config = self.config['training'].get('save_checkpoint_path')
        if not save_path_config or save_path_config.lower() == 'null':
            self.logger.info("--- 未指定保存路径, 跳过保存检查点 ---")
            return
        final_save_path = os.path.join(self.project_root, save_path_config)
        save_dir = os.path.dirname(final_save_path)
        os.makedirs(save_dir, exist_ok=True)
        try:
            self.logger.info(f"--- 正在保存【完整】训练检查点 (用于 cycle {cycle_num}) ---")
            self.logger.info(f"路径: {final_save_path}")
            cleaned_replay_buffer = []
            for (tensor_norm, score, label) in self.replay_buffer:
                cleaned_replay_buffer.append(
                    (tensor_norm.cpu().detach(), score, label.cpu().detach())
                )
            torch.save({
                'model_state_dict': self.diffusion_model.state_dict(),
                'rl_agent_state_dict': self.rl_agent.state_dict(),
                'dit_optimizer_state_dict': self.dit_optimizer.state_dict(),
                'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
                'cycle': cycle_num,
                'replay_buffer': cleaned_replay_buffer
            }, final_save_path)
            self.logger.info("--- 完整检查点保存成功 ---")
            self.logger.info("\n")
        except Exception as e:
            self.logger.error(f"[错误] 保存检查点失败: {e}")
            self.logger.info("\n")