# src/pipeline.py

import os
import json
import logging
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

# 导入所有模块
from .utils import dataloader
from .evaluator.evaluator import MechanismEvaluator
from .diffusion_model.model import DiffusionModel
from .rl_agent.agent import RLAgent
from .solver.utils import tensor_to_graph, find_independent_loops
from .solver.simple_kinematics import compute_loop_errors, compute_bennett_geometry_error


class TrainingPipeline:
    def __init__(self, config, project_root):
        # --- 获取 Logger ---
        self.logger = logging.getLogger()

        self.config = config
        self.project_root = project_root

        # --- 1. 初始化硬件 ---
        self.device = self._setup_device()

        # --- 2. 加载数据归一化参数和生成参数 ---
        try:
            norm_values_dict = config['data']['normalization_values']
            # (新!) 归一化向量仅用于 [a, alpha, offset]
            # (我们假设 config 中的 'd' 键对应 'offset' 通道)
            norm_vec = torch.tensor([
                norm_values_dict['a'],
                norm_values_dict['alpha'],
                norm_values_dict['d']  # 'd' 键用于 offset
            ], dtype=torch.float32, device=self.device)
            self.norm_vec = norm_vec.view(1, 3, 1, 1)  # (1, 3, 1, 1)
        except KeyError:
            raise ValueError("[致命错误] 配置文件中缺少 data.normalization_values 块 (需要 a, alpha, d)。")

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
        self.start_cycle = 0  # (新!) 默认从 cycle 0 开始
        self.replay_buffer = []  # (新!) 默认 Replay Buffer 为空

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
                        # 加载已完成的 cycle 编号, 下一次从 +1 开始
                        self.start_cycle = checkpoint['cycle'] + 1
                        self.logger.info(
                            f"  > 训练进度... OK (将从 cycle {self.start_cycle} / {self.config['training']['num_cycles']} 处恢复)")

                    if 'replay_buffer' in checkpoint:
                        loaded_replay_buffer = checkpoint['replay_buffer']

                        # --- 将加载的 CPU Tensors 移回 GPU (self.device) ---
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
            # 如果没有加载检查点(或失败了), 则执行标准启动流程
            self.logger.info("--- 未指定或加载检查点失败, 所有模块使用随机初始化权重 ---")
            # 仅在此时才从 augmented_dataset 加载 Replay Buffer
            self.replay_buffer = self.load_replay_buffer()

        # --- 6. 加载实验开关 (位置不变) ---
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
        """设置训练设备 (GPU 或 CPU)"""
        if self.config['training'].get('device', 'cpu') == 'cuda' and torch.cuda.is_available():
            self.logger.info("训练将在 NVIDIA CUDA GPU 上运行。")
            return torch.device("cuda")
        else:
            self.logger.info("训练将在 CPU 上运行。")
            return torch.device("cpu")

    # --- 归一化/反归一化 辅助函数 ---
    # (模型内部有自己的版本, 但 pipeline 在处理 Replay Buffer 时也需要)
    def _normalize(self, x_tensor):
        """ (新!) 将 (B, 5, H, W) 或 (C, 5, H, W) 数据按通道归一化到 [-1, 1]."""
        original_dim = x_tensor.dim()
        if original_dim == 3:
            x_tensor = x_tensor.unsqueeze(0);
            was_3d = True
        else:
            was_3d = False

        if x_tensor.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor.shape[1]}")

        # --- 关键修正: 提前将整个张量移动到设备 ---
        x_tensor = x_tensor.to(self.device)

        exists_channel = x_tensor[:, 0:1, :, :]  # (B, 1, H, W) [exists]
        joint_type_channel = x_tensor[:, 1:2, :, :]  # (B, 1, H, W) [joint_type]
        other_channels = x_tensor[:, 2:, :, :]  # (B, 3, H, W) [a, alpha, offset]

        exists_norm = exists_channel * 2.0 - 1.0

        # joint_type 已经是 -1/1, 无需操作
        joint_type_norm = joint_type_channel

        # self.norm_vec 已经在 __init__ 中被移动到了 self.device
        # (B, 3, H, W) / (1, 3, 1, 1)
        other_norm = (other_channels / (self.norm_vec + 1e-8)) * 2.0 - 1.0

        result = torch.cat([exists_norm, joint_type_norm, other_norm], dim=1)

        if was_3d: return result.squeeze(0)
        return result

    def _unnormalize(self, x_tensor_norm):
        """ (新!) 将 (B, 5, H, W) 或 (C, 5, H, W) 数据从 [-1, 1] 按通道恢复."""
        original_dim = x_tensor_norm.dim()
        if original_dim == 3:
            x_tensor_norm = x_tensor_norm.unsqueeze(0);
            was_3d = True
        else:
            was_3d = False

        if x_tensor_norm.shape[1] != 5: raise ValueError(f"通道数应为5, 收到 {x_tensor_norm.shape[1]}")

        # --- 关键修正: 提前将整个张量移动到设备 ---
        x_tensor_norm = x_tensor_norm.to(self.device)

        exists_norm = x_tensor_norm[:, 0:1, :, :]  # (B, 1, H, W)
        joint_type_norm = x_tensor_norm[:, 1:2, :, :]  # (B, 1, H, W)
        other_norm = x_tensor_norm[:, 2:, :, :]  # (B, 3, H, W)

        exists_unnorm = (exists_norm + 1.0) / 2.0

        # joint_type 已经是 -1/1, 无需操作
        joint_type_unnorm = joint_type_norm

        # self.norm_vec 已经在 GPU 上
        other_unnorm = ((other_norm + 1.0) / 2.0) * self.norm_vec

        result = torch.cat([exists_unnorm, joint_type_unnorm, other_unnorm], dim=1)

        if was_3d: return result.squeeze(0)
        return result

    # --- 加载检查点 ---
    # (此方法已在 __init__ 中集成，为了保持原文件结构，如果下面有被调用可以保留，否则可删除，这里为了稳妥不重复定义)

    def load_replay_buffer(self):
        """
        (已更新!) 从 augmented_dataset 加载经验。
        (新!) 自动检测 6 列 (旧) 和 8 列 (新) 的 .npz 格式.
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
        label_map = {"bennett": 0}  # (保持简单)

        for entry in manifest:
            npz_path = os.path.join(data_dir, entry['data_path'])
            try:
                # --- 检查标签 ---
                label_str = entry.get('metadata', {}).get('label')
                if label_str is None:
                    self.logger.warning(f"警告: 条目 {entry['id']} 缺少标签, 跳过。")
                    continue
                label_idx = label_map.get(label_str, -1)
                if label_idx == -1:
                    self.logger.warning(f"警告: 条目 {entry['id']} 标签 '{label_str}' 未知, 跳过。")
                    continue

                with np.load(npz_path) as data:
                    if 'edge_list_array' not in data:
                        self.logger.warning(f"警告: {npz_path} 中未找到 'edge_list_array', 跳过。")
                        continue

                    edge_list_array = data['edge_list_array']

                    # 像 Dataloader 一样动态构建稠密张量
                    max_nodes = self.config['data']['max_nodes']
                    # (新!) 通道数更新为 5
                    num_features = self.config['diffusion_model']['in_channels']
                    feature_tensor = np.zeros((max_nodes, max_nodes, num_features), dtype=np.float32)

                    num_cols = edge_list_array.shape[1] if edge_list_array.ndim == 2 else 0

                    for edge_data in edge_list_array:
                        # --- (新!) 6 列/ 8 列 兼容逻辑 ---
                        if num_cols == 6:
                            # (旧格式) [i, k, a, alpha, offset_i, offset_k]
                            i, k, a, alpha, offset_i, offset_k = edge_data
                            joint_type_i = 1.0  # 假定 R 副
                            joint_type_k = 1.0  # 假定 R 副
                        elif num_cols == 8:
                            # (新格式) [i, k, a, alpha, offset_i, offset_k, joint_type_i, joint_type_k]
                            i, k, a, alpha, offset_i, offset_k, joint_type_i, joint_type_k = edge_data
                        else:
                            continue  # 格式错误

                        i, k = int(i), int(k)
                        if i >= max_nodes or k >= max_nodes: continue

                        # 填充 5 个通道: [exists, joint_type, a, alpha, offset]
                        # 注意：joint_type 应该是节点属性，但在 edge_list 中可能存储在边上
                        # 这里我们直接填入
                        feature_tensor[i, k] = [1.0, joint_type_i, a, alpha, offset_i]
                        feature_tensor[k, i] = [1.0, joint_type_k, a, alpha, offset_k]

                    # (H, W, C) -> (C, H, W)
                    tensor_unnorm_torch = torch.from_numpy(np.transpose(feature_tensor, (2, 0, 1))).float()

                    # (新!) _normalize 现在处理 5 通道
                    tensor_norm = self._normalize(tensor_unnorm_torch)
                    score = entry.get('metadata', {}).get('score', 0.0)

                    experiences.append((tensor_norm, score, torch.tensor(label_idx, dtype=torch.long)))

            except Exception as e:
                self.logger.warning(f"警告: 无法加载 Replay Buffer 条目 {npz_path}. 原因: {e}")

        self.logger.info(f"Replay Buffer 加载完成. 共 {len(experiences)} 个历史经验。")
        return experiences

    def _generate_and_augment(self, cycle_num, total_cycles):
        """
        步骤 1 & 2: 生成, 筛选, 软优化(S+Q), 评估, 扩充.
        (使用纯 PyTorch 软约束优化，替代 Solver)
        """
        self.logger.info("--- 步骤 1 & 2: 生成, 筛选, 软优化, 扩充 ---")
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

        eval_config = self.config.get('evaluator_config', {})

        # A. Bennett 配置
        label_indicators = eval_config.get('label_specific_indicators', {}).get(current_target_label_str, {})
        bennett_conf = label_indicators.get('check_is_bennett', {})
        enable_bennett_opt = bennett_conf.get('enable', False)
        # 提前计算权重 (如果没配 weight 默认为 1.0，再乘以 20.0 的强化系数)
        bennett_opt_weight = bennett_conf.get('weight', 1.0) * 20.0

        # B. 闭环配置
        common_indicators = eval_config.get('common_indicators', {})
        closure_conf = common_indicators.get('check_kinematic_feasibility', {})
        enable_closure_opt = closure_conf.get('enable', True)

        # C. 优化容忍度
        opt_tolerance = self.config['generation'].get('optimization_tolerance', 0.5)

        self.logger.info(f"优化器配置: 闭环={enable_closure_opt}, "
                         f"Bennett几何={enable_bennett_opt} (Weight={bennett_opt_weight}), "
                         f"容忍度={opt_tolerance}")

        self.logger.info(f"正在生成 {num_to_gen} 个目标标签为 '{current_target_label_str}' 的机构...")

        # 1. DiT 采样 (获取原始 Tensor)
        _, batch_x0_norm_pure = self.diffusion_model.sample(
            num_samples=num_to_gen,
            y=target_labels,
            guidance_fn=guidance_fn,
            guidance_scale=self.guidance_scale
        )

        # ==================================================
        # 2. --- 拓扑筛选与软优化 (Topology Filter & Soft Opt) ---
        # ==================================================
        self.logger.info("正在进行拓扑筛选与软约束优化...")

        new_experiences_for_rl = []
        good_mechanisms_to_save = []
        num_satisfying_score = 0

        # 预处理：转 Numpy 进行图分析
        # 注意：这里只是临时转一下用来找环，优化还是用 Tensor
        x_phys_temp = self._unnormalize(batch_x0_norm_pure).detach()  # (B, 5, N, N)

        # 逐个样本处理 (因为优化过程是针对每个机构独立的)
        for i in range(num_to_gen):
            # --- A. 拓扑筛选 ---
            # 取出当前样本的结构 (N, N, 5)
            struct_tensor = x_phys_temp[i].permute(1, 2, 0)

            # 使用 solver.utils 工具建图
            G = tensor_to_graph(struct_tensor)
            loops = find_independent_loops(G)

            # 只有拓扑合格（有环）的机构才值得优化
            if not loops:
                # 没环的直接跳过，不放入 RL 经验池（或者放入低分经验）
                # 这里选择放入低分经验，让 RL 知道生成无环是不好的
                score = -1.0
                new_experiences_for_rl.append((batch_x0_norm_pure[i], score, target_labels[i]))
                continue

            # ==============================================================
            # [新增] 2. 检查环路长度 (必须 >= 4)
            # 三角形 (len=3) 在机构学上是刚性结构，没有任何意义
            # ==============================================================
            if len(loops[0]) < 4:
                # print(f"  > Mech {i+1:02d}: 跳过 (拓扑退化为 {len(loops[0])} 杆结构)")
                score = -1.0
                new_experiences_for_rl.append((batch_x0_norm_pure[i], score, target_labels[i]))
                continue

            # --- B. 软约束优化 (Soft Optimization) ---
            # 目标：同时微调 结构(S) 和 关节(Q) 使得闭环误差最小

            # 1. 准备优化变量
            # x_opt: 结构参数 (从 DiT 输出初始化)
            x_opt = batch_x0_norm_pure[i].unsqueeze(0).detach().clone().requires_grad_(True)

            # q_opt: 关节参数 (随机初始化)
            # 初始化在 [-0.5, 0.5] 弧度之间，不要太大
            q_opt = (torch.randn(self.config['data']['max_nodes'], device=self.device) * 0.5).requires_grad_(True)

            # ==========================================================
            # [核心修正] 强制“硬复位”：在优化开始前，把数值强行按回 [-1, 1]
            # ==========================================================
            with torch.no_grad():
                # 1. 强制所有通道归一化值在 [-1, 1] 之间
                # 这样解归一化后的物理值绝不会超过 config 中定义的最大值 (如 20.0)
                x_opt.clamp_(-1.0, 1.0)

                # 2. (可选) 更进一步，把几何参数 (通道2,3,4) 初始化得更小一点
                # 相当于告诉优化器：“先别管 DiT 生成的长短，先从短杆开始修”
                # 通道: 0:exists, 1:type, 2:a, 3:alpha, 4:offset
                # 我们把 a 和 offset (归一化后) 缩小到 [-0.1, 0.1] 附近
                # 这样初始物理长度大约是 2.0 左右，误差大概在几十，而不是几十亿
                x_opt[:, 2:] = x_opt[:, 2:] * 0.1

            # 2. 定义优化器 (Adam)
            # 结构参数 LR 小一点(微调)，关节参数 LR 大一点(寻找解)
            optimizer = optim.Adam([
                {'params': x_opt, 'lr': 0.005},
                {'params': q_opt, 'lr': 0.05}
            ])

            best_loss_for_this_mech = float('inf')
            best_x_final = x_opt.detach().clone()
            best_q_final = None

            # [打印] 记录初始状态 (为了对比优化效果)
            initial_loss_val = float('inf')

            # 3. 优化循环 (200步)
            for step in range(200):
                optimizer.zero_grad()

                # --- A. 正向计算 ---
                x_phys = self.diffusion_model.apply_physics_constraints(x_opt, clamp=False)
                structure = x_phys[0].permute(1, 2, 0)

                # --- B. 动态计算 Loss ---
                loss = 0.0

                # 直接使用循环外定义好的开关变量
                if enable_closure_opt:
                    loss_closure = compute_loop_errors(structure, q_opt, loops)
                    loss += loss_closure
                else:
                    loss_closure = torch.tensor(0.0)

                if enable_bennett_opt:
                    loss_bennett_geo = compute_bennett_geometry_error(structure, loops)
                    # 直接使用循环外计算好的权重
                    loss += bennett_opt_weight * loss_bennett_geo
                else:
                    loss_bennett_geo = torch.tensor(0.0)

                # 3. 边界与正则 Loss (始终开启，防止数值爆炸)
                _, _, a, alpha, offset = torch.chunk(x_phys, 5, dim=1)

                loss_bounds = 0.0
                loss_bounds += torch.relu(-a).sum()
                loss_bounds += torch.relu(0.5 - a).sum() * 50.0  # 防退化
                loss_bounds += torch.relu(torch.abs(offset) - 25.0).sum()

                reg_loss = 0.01 * (q_opt ** 2).mean() + 0.01 * (offset ** 2).mean()

                loss += 10.0 * loss_bounds + 1.0 * reg_loss

                # --- C. 记录最优解 ---
                # 此时的 loss_closure 和 loss_bennett_geo 都是计算好的
                # 我们用它们的和作为衡量标准 (如果开启的话)

                current_metric = 0.0
                if enable_closure_opt: current_metric += loss_closure.item()
                if enable_bennett_opt: current_metric += loss_bennett_geo.item()

                if step == 0: initial_loss_val = current_metric

                if current_metric < best_loss_for_this_mech:
                    best_loss_for_this_mech = current_metric
                    best_x_final = x_opt.detach().clone()
                    best_q_final = q_opt.detach().clone()

                # --- D. 更新 ---
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x_opt, q_opt], max_norm=1.0)
                optimizer.step()

                # --- E. 投影修正 ---
                with torch.no_grad():
                    x_opt.data.clamp_(-1.0, 1.0)
                    x_opt.data[:, 2, :, :].clamp_(min=-0.95)
                    q_opt.data.clamp_(-math.pi, math.pi)

                # --- F. 调试打印 ---
                if i == 0 and (step == 0 or (step + 1) % 10 == 0):
                    print(f"  [Debug Mech 0] Step {step + 1:02d}/50 | "
                          f"Total: {loss.item():.2f} | "
                          f"Metric: {current_metric:.4f} | "
                          f"GeoLoss: {loss_bennett_geo.item():.4f}")

            # --- [新增] 打印每个机构的最终优化结果 ---
            status = "OK" if best_loss_for_this_mech < opt_tolerance else "FAIL"

            # 修正逻辑：如果是第1个，或者整除5，或者状态是 OK，都打印
            # 这样保证了每一个成功的机构都会先打印这一行 summary，再打印下面的详细参数
            if (i + 1) % 5 == 0 or i == 0 or status == "OK":
                print(f"  > Mech {i + 1:02d}/{num_to_gen}: {status} "
                      f"(Loss: {initial_loss_val:.4f} -> {best_loss_for_this_mech:.6f})")

            # --- C. 评估与保存 ---

            final_phys = self.diffusion_model.apply_physics_constraints(best_x_final, clamp=True).detach()
            final_np = final_phys[0].permute(1, 2, 0).cpu().numpy()

            if best_loss_for_this_mech < opt_tolerance:
                score = self.evaluator.evaluate(
                    final_np,
                    target_label=current_target_label_str,
                    current_cycle=cycle_num,
                    total_cycles=total_cycles,
                    optimized_joint_angles=best_q_final,
                    known_loops=loops
                )
                # =========================================================
                # [新增] 打印详细参数 (包含 Joint Type 和 DH参数 d)
                # =========================================================
                print(f"\n  >>> [详细参数报告] Mech {i + 1:02d} (Score: {score:.4f}) <<<")

                # 1. 打印非结构参数 (关节变量 q)
                q_vals = best_q_final.detach().cpu().numpy()

                # 2. 打印结构参数 (沿着环路打印)
                if loops and len(loops) > 0:
                    path = loops[0]  # 取主环路
                    print(f"  环路路径: {path}")

                    L = len(path)
                    for idx, node_u in enumerate(path):
                        node_v = path[(idx + 1) % L]  # 下一个节点 (出边)
                        node_prev = path[(idx - 1 + L) % L]  # 上一个节点 (入边)

                        # 获取参数
                        # final_np shape: (N, N, 5) -> [u, v, :]
                        params = final_np[node_u, node_v]

                        j_type_val = params[1]
                        type_str = "R" if j_type_val > 0 else "P"

                        a_val = params[2]
                        alpha_val = params[3]

                        # --- [核心修改] 计算 DH 参数 d (相对偏移量) ---
                        # offset_out: 当前关节 u 连接到下一关节 v 的接口位置
                        offset_out = params[4]
                        # offset_in:  当前关节 u 连接到上一关节 prev 的接口位置
                        offset_in = final_np[node_u, node_prev, 4]

                        # d = out - in
                        d_val = offset_out - offset_in

                        # 获取关节变量
                        q_val = q_vals[node_u]

                        # 打印: 显示计算后的 d 而不是原始 offset
                        print(f"    [Joint {node_u} | {type_str}] q={q_val:8.4f}  -->  "
                              f"[Link {node_u}-{node_v}] a={a_val:8.4f}, "
                              f"alpha={alpha_val:8.4f}, d={d_val:8.4f}")  # 这里打印 d
                else:
                    print("  (警告: 未找到有效环路，无法打印连杆参数)")
                print("  " + "-" * 60 + "\n")
            else:
                score = -1.0

            # --- [修正 2] 使用 best_x_final ---
            new_experiences_for_rl.append(
                (best_x_final[0], score, target_labels[i])
            )

            if score >= self.acceptance_threshold:
                num_satisfying_score += 1
                if self.enable_augmentation:
                    new_entry = {
                        "tensor": final_np,
                        "metadata": {
                            "source": "soft_optimized",
                            "generation_cycle": cycle_num + 1,
                            "score": score,
                            "label": current_target_label_str,
                            # --- [修正 3] 使用 best_loss_for_this_mech ---
                            "closure_error": best_loss_for_this_mech
                        }
                    }
                    good_mechanisms_to_save.append(new_entry)

        # ==================================================
        # 3. 日志与保存逻辑 (保持原样)
        # ==================================================

        if new_experiences_for_rl:
            all_scores = [exp[1] for exp in new_experiences_for_rl]
            avg_score = sum(all_scores) / len(all_scores)
            min_score = min(all_scores)
            max_score = max(all_scores)
            self.logger.info(f"生成机构平均得分: {avg_score:.4f} (Min: {min_score:.4f}, Max: {max_score:.4f})")
        else:
            avg_score = None

        self.logger.info(
            f"评估完成. {num_satisfying_score} / {num_to_gen} 个机构满足分数要求...")

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
            self.replay_buffer,
            self.diffusion_model,
            self.rl_optimizer,
            self.device
        )

        if avg_rl_loss is not None:
            self.logger.info(f"--- RL Agent 训练完成. 平均 Loss: {avg_rl_loss:.6f} ---")

    def _train_dit_model(self, cycle_num):
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
            for x_start, y_labels in current_dataloader:
                x_start = x_start.to(self.device)
                y_labels = y_labels.to(self.device)
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                x_start_norm = self._normalize(x_start)
                noise = torch.randn_like(x_start_norm)
                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)

                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)

                loss = F.mse_loss(predicted_noise_norm, noise)
                loss.backward()
                self.dit_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(current_dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"  DiT Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

    def _warmup_dit(self, num_epochs):
        self.logger.info(f"--- DiT 预热阶段: 在初始数据集上训练 {num_epochs} 轮 ---")

        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])
        warmup_dataloader = dataloader.get_dataloader(
            self.config,
            initial_manifest_path,
            None,
            self.config['training']['batch_size'],
            shuffle=True
        )

        if warmup_dataloader is None or len(warmup_dataloader.dataset) == 0:
            self.logger.warning("[警告] 初始数据集为空, 跳过预热阶段。")
            return

        self.logger.info(f"初始数据集大小: {len(warmup_dataloader.dataset)} 个机构")

        self.diffusion_model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_start, y_labels in warmup_dataloader:
                x_start = x_start.to(self.device)
                y_labels = y_labels.to(self.device)
                batch_size = x_start.size(0)

                self.dit_optimizer.zero_grad()
                t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=self.device).long()

                x_start_norm = self._normalize(x_start)
                noise = torch.randn_like(x_start_norm)
                x_t_norm = self.diffusion_model.q_sample(x_start_norm, t, noise)
                predicted_noise_norm = self.diffusion_model(x_t_norm, t, y_labels)
                loss = F.mse_loss(predicted_noise_norm, noise)

                loss.backward()
                self.dit_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(warmup_dataloader)

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"  预热 Epoch [{epoch + 1:03d}/{num_epochs:03d}] | MSE Loss: {avg_loss:.6f}")

        self.logger.info("--- DiT 预热完成! 模型已学习初始数据集的基本结构 ---")

    def _warmup_rl_agent(self, num_epochs):
        self.logger.info(f"--- RL 智能体预热阶段: 在初始数据集上训练 {num_epochs} 轮 ---")

        initial_manifest_path = os.path.join(self.project_root, self.config['data']['initial_manifest_path'])
        initial_dataset = dataloader.MechanismDataset(
            self.config,
            initial_manifest_path,
            None
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
                x_start_unnorm = x_start_batch[i]
                y_label = y_labels_batch[i]

                x_start_norm = self._normalize(x_start_unnorm)
                x_start_unnorm_numpy = x_start_unnorm.permute(1, 2, 0).cpu().numpy()
                score = self.evaluator.evaluate(x_start_unnorm_numpy)

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
        """
        执行完整的三步循环：生成 -> 训练RL -> 训练DiT
        包含完整的预热逻辑和日志记录。
        """
        # --- DiT 预热阶段 ---
        warmup_epochs = self.config['training'].get('dit_warmup_epochs', 0)
        if warmup_epochs > 0 and not self.checkpoint_loaded:
            self.logger.info("[检测到] DiT 使用随机初始化权重, 将进行预热训练...")
            self._warmup_dit(warmup_epochs)
        elif self.checkpoint_loaded:
            self.logger.info(f"[跳过预热] DiT 已从检查点加载 (将从 cycle {self.start_cycle} 开始)。")
        else:
            self.logger.info("[跳过预热] 配置文件中 dit_warmup_epochs=0, 不进行预热。")

        # --- RL Agent 预热阶段 ---
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

        # --- 主训练循环 ---
        num_cycles = self.config['training']['num_cycles']
        self.logger.info(f"--- 开始总共 {num_cycles} 轮的训练循环 ---")

        for cycle in range(self.start_cycle, num_cycles):
            self.logger.info(f"===== [ 完整循环 {cycle + 1}/{num_cycles} ] =====")
            self.logger.info("\n")

            # 步骤 1 & 2: 生成, 修复, 评估, 扩充
            start_gen_eval = time.time()
            new_experiences = self._generate_and_augment(cycle, num_cycles)
            self.logger.info(f"  生成与评估耗时: {time.time() - start_gen_eval:.2f} 秒")
            self.logger.info("\n")

            # 步骤 3a: 训练 RL 智能体
            start_rl_train = time.time()
            self._train_rl_agent(new_experiences)
            self.logger.info(f"  RL 训练耗时: {time.time() - start_rl_train:.2f} 秒")
            self.logger.info("\n")

            # 步骤 3b: 训练 DiT 模型
            start_dit_train = time.time()
            self._train_dit_model(cycle)
            self.logger.info(f"  DiT 训练耗时: {time.time() - start_dit_train:.2f} 秒")
            self.logger.info("\n")

            # 保存
            self.save_checkpoint(cycle_num=cycle)

        self.logger.info("===== 所有训练循环完成! =====")
        self.save_checkpoint(cycle_num=num_cycles)

    def save_checkpoint(self, cycle_num=0):
        """
        保存完整的训练状态, 包括 DiT, RL Agent, 优化器, 和训练进度.
        在保存 Replay Buffer 之前将其“清理”到 CPU.
        """
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

            # --- "清理" Replay Buffer 以便安全保存 ---
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