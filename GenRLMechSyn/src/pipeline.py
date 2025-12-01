# src/pipeline.py

import os
import json
import logging
import time
import math                 # [新增]
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networkx as nx       # [新增]

# 导入所有模块
from .utils import dataloader
from .evaluator.evaluator import MechanismEvaluator
from .diffusion_model.model import DiffusionModel
from .rl_agent.agent import RLAgent
from .solver.utils import tensor_to_graph, find_independent_loops

from .solver.simple_kinematics import (
    compute_loop_errors,
    compute_bennett_geometry_error,
    # compute_all_joint_screws, # 这个如果pipeline里不直接用可以去掉
    compute_mobility_loss_eigen,
    compute_task_loss_eigen,
    compute_motion_consistency_loss
)


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
        从 augmented_dataset 加载经验。
        自动检测 6 列 (旧) 和 8 列 (新) 的 .npz 格式.
        [修复] 从 config 读取 label_mapping，解决 'general_2dof' 未知错误。
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

        # 从配置读取标签映射，而不是硬编码
        # 这样 pipeline 就能认识 'general_2dof' 了
        label_map = self.config['data'].get('label_mapping', {"bennett": 0})

        for entry in manifest:
            npz_path = os.path.join(data_dir, entry['data_path'])
            try:
                # --- 检查标签 ---
                label_str = entry.get('metadata', {}).get('label')
                if label_str is None:
                    # 兼容旧数据，如果没标签默认当 bennett
                    label_str = "bennett"

                label_idx = label_map.get(label_str, -1)
                if label_idx == -1:
                    self.logger.warning(f"警告: 条目 {entry['id']} 标签 '{label_str}' 未知 (Config中未定义), 跳过。")
                    continue

                with np.load(npz_path) as data:
                    if 'edge_list_array' not in data:
                        self.logger.warning(f"警告: {npz_path} 中未找到 'edge_list_array', 跳过。")
                        continue

                    edge_list_array = data['edge_list_array']

                    # 像 Dataloader 一样动态构建稠密张量
                    max_nodes = self.config['data']['max_nodes']
                    num_features = self.config['diffusion_model']['in_channels']
                    feature_tensor = np.zeros((max_nodes, max_nodes, num_features), dtype=np.float32)

                    num_cols = edge_list_array.shape[1] if edge_list_array.ndim == 2 else 0

                    for edge_data in edge_list_array:
                        # 兼容 6 列/ 8 列
                        if num_cols == 6:
                            i, k, a, alpha, offset_i, offset_k = edge_data
                            joint_type_i, joint_type_k = 1.0, 1.0
                        elif num_cols == 8:
                            i, k, a, alpha, offset_i, offset_k, joint_type_i, joint_type_k = edge_data
                        else:
                            continue

                        i, k = int(i), int(k)
                        if i >= max_nodes or k >= max_nodes: continue

                        feature_tensor[i, k] = [1.0, joint_type_i, a, alpha, offset_i]
                        feature_tensor[k, i] = [1.0, joint_type_k, a, alpha, offset_k]

                    tensor_unnorm_torch = torch.from_numpy(np.transpose(feature_tensor, (2, 0, 1))).float()
                    tensor_norm = self._normalize(tensor_unnorm_torch)
                    score = entry.get('metadata', {}).get('score', 0.0)

                    experiences.append((tensor_norm, score, torch.tensor(label_idx, dtype=torch.long)))

            except Exception as e:
                self.logger.warning(f"警告: 无法加载 Replay Buffer 条目 {npz_path}. 原因: {e}")

        self.logger.info(f"Replay Buffer 加载完成. 共 {len(experiences)} 个历史经验。")
        return experiences

    def _generate_and_augment(self, cycle_num, total_cycles):
        """
        步骤 1 & 2: 生成, 筛选, 软优化, 扩充.
        (完整版 v2.6: 模块化重构 - 调用 _optimize_mechanism)
        """
        self.logger.info("--- 步骤 1 & 2: 生成, 筛选, 软优化, 扩充 ---")
        self.diffusion_model.eval()
        self.rl_agent.eval()

        # 1. 引导设置
        guidance_fn = None
        if self.enable_rl_guidance:
            guidance_fn = self.rl_agent.get_guidance_fn(self.guidance_scale)

        # 2. 生成配置
        num_to_gen = self.config['generation']['num_to_generate']
        target_labels = torch.full((num_to_gen,), self.target_label_index, dtype=torch.long, device=self.device)

        label_map = self.config['data'].get('label_mapping', {"bennett": 0})
        idx_to_label = {v: k for k, v in label_map.items()}
        target_str = idx_to_label.get(self.target_label_index, "unknown")

        self.logger.info(f"正在生成 {num_to_gen} 个 '{target_str}' 机构...")

        # 3. DiT 采样
        _, batch_x0_norm_pure = self.diffusion_model.sample(
            num_samples=num_to_gen, y=target_labels,
            guidance_fn=guidance_fn, guidance_scale=self.guidance_scale
        )

        # ==========================================================
        # 4. 解析优化配置 (打包参数传递给子函数)
        # ==========================================================
        gen_conf = self.config['generation']
        eval_conf = self.config.get('evaluator_config', {})
        common_conf = eval_conf.get('common_indicators', {})

        # A. 开关配置字典
        opt_config = {
            'enable_closure': common_conf.get('check_kinematic_feasibility', {}).get('enable', True),
            'enable_mobility': common_conf.get('check_mobility', {}).get('enable', True),
            'enable_task': common_conf.get('check_task_performance', {}).get('enable', False),
            'enable_consistency': common_conf.get('check_global_consistency', {}).get('enable', False),
            'enable_bennett': eval_conf.get('label_specific_indicators', {}).get(target_str, {}).get('check_is_bennett',
                                                                                                     {}).get('enable',
                                                                                                             False),
            'num_restarts': gen_conf.get('num_restarts', 5),
            'tolerance': gen_conf.get('optimization_tolerance', 0.5),
            'ee_node': gen_conf.get('ee_node', self.config['data']['max_nodes'] - 1),
            'gap_threshold': common_conf.get('check_mobility', {}).get('gap_threshold', 0.005)
        }

        # B. 权重配置字典
        weights = {
            'mobility': common_conf.get('check_mobility', {}).get('weight', 1.0),
            'task': common_conf.get('check_task_performance', {}).get('weight', 5.0),
            'consistency': common_conf.get('check_global_consistency', {}).get('weight', 2.0),
            'bennett': eval_conf.get('label_specific_indicators', {}).get(target_str, {}).get('check_is_bennett',
                                                                                              {}).get('weight',
                                                                                                      1.0) * 20.0
        }

        # C. 任务数据字典
        target_data = {'num_dof': 1, 'twists': None, 'masks': None}
        if opt_config['enable_task']:
            raw_patterns = common_conf.get('check_task_performance', {}).get('target_motion_patterns', [])
            if raw_patterns:
                target_data['num_dof'] = len(raw_patterns)
                twists = torch.zeros((len(raw_patterns), 6), device=self.device)
                masks = torch.zeros((len(raw_patterns), 6), device=self.device)
                for k, pat in enumerate(raw_patterns):
                    for d, val in enumerate(pat):
                        if val is not None:
                            twists[k, d] = float(val);
                            masks[k, d] = 1.0
                target_data['twists'] = twists
                target_data['masks'] = masks

        self.logger.info(f"优化配置: {opt_config}")

        # ==========================================================
        # 5. 逐个处理循环
        # ==========================================================
        new_experiences = []
        good_mechanisms = []
        num_ok = 0
        x_phys_temp = self._unnormalize(batch_x0_norm_pure).detach()

        for i in range(num_to_gen):
            # A. 拓扑筛选
            struct_tsr = x_phys_temp[i].permute(1, 2, 0)
            G = tensor_to_graph(struct_tsr)
            loops = find_independent_loops(G)

            has_valid_loop = False
            if loops:
                for loop in loops:
                    if len(loop) >= 4: has_valid_loop = True; break

            if not has_valid_loop:
                new_experiences.append((batch_x0_norm_pure[i], -1.0, target_labels[i]))
                continue

            # B. 调用独立优化函数 [核心]
            x_init = batch_x0_norm_pure[i].unsqueeze(0).detach()

            best_x, best_q, best_metric, init_metric = self._optimize_mechanism(
                i, x_init, loops, G, opt_config, weights, target_data
            )

            # C. 结果判定
            is_ok = best_metric < opt_config['tolerance']
            status = "OK" if is_ok else "FAIL"

            if (i + 1) % 5 == 0 or i == 0 or is_ok:
                print(f"  > Mech {i + 1:02d}: {status} (Metric: {init_metric:.4f} -> {best_metric:.6f})")

            # D. 评估与打印
            final_phys = self.diffusion_model.apply_physics_constraints(best_x, clamp=True).detach()
            final_np = final_phys[0].permute(1, 2, 0).cpu().numpy()

            score = -1.0
            if is_ok:
                score = self.evaluator.evaluate(
                    final_np, target_label=target_str, current_cycle=cycle_num,
                    total_cycles=total_cycles, optimized_joint_angles=best_q, known_loops=loops
                )

                # 打印详细报告 (这里可以保持原样，或进一步封装成 helper)
                print(f"\n  >>> [详细参数报告] Mech {i + 1:02d} (Score: {score:.4f}) <<<")
                q_vals = best_q.detach().cpu().numpy()
                if loops:
                    for loop_idx, path in enumerate(loops):
                        l_type = "Rigid" if len(path) < 4 else "Kinematic"
                        print(f"  --- Loop {loop_idx + 1}: {path} ({l_type}) ---")
                        L = len(path)
                        for idx, u in enumerate(path):
                            v = path[(idx + 1) % L];
                            prev = path[(idx - 1 + L) % L]
                            p = final_np[u, v]
                            t_str = "R" if p[1] > 0 else "P"
                            d_val = p[4] - final_np[u, prev, 4]
                            print(f"    [{u}|{t_str}] q={q_vals[u]:.4f} --> a={p[2]:.4f}, al={p[3]:.4f}, d={d_val:.4f}")
                print("-" * 60 + "\n")

            # E. 收集结果
            new_experiences.append((best_x[0], score, target_labels[i]))
            if score >= self.acceptance_threshold:
                num_ok += 1
                if self.enable_augmentation:
                    good_mechanisms.append({
                        "tensor": final_np,
                        "metadata": {"source": "soft_optimized", "score": score, "label": target_str,
                                     "closure_error": best_metric}
                    })

        # F. 结束统计
        if new_experiences:
            scores = [e[1] for e in new_experiences]
            if scores:
                avg = sum(scores) / len(scores)
                self.logger.info(f"生成机构平均得分: {avg:.4f} (Min: {min(scores):.4f}, Max: {max(scores):.4f})")

        self.logger.info(f"评估完成. {num_ok} / {num_to_gen} 合格.")

        if self.enable_augmentation and good_mechanisms:
            self.logger.info(f"数据增强: 保存 {len(good_mechanisms)} 个新机构。")
            dataloader.add_mechanisms_to_dataset(good_mechanisms,
                                                 os.path.join(self.project_root,
                                                              self.config['data']['augmented_manifest_path']))

        return new_experiences

    def _optimize_mechanism(self, mech_idx, x_init_raw, loops, G,
                            opt_config, weights, target_data):
        """
        封装单体机构的完整优化过程 (Multi-Start + Gradient Descent)。
        包含：初始化、正向计算、Loss(闭环/Bennett/可动性/任务/一致性)、反向传播、投影梯度。

        Args:
            mech_idx: 当前机构的索引 (用于控制日志打印频率)
            x_init_raw: DiT 生成的原始结构张量 (未归一化)
            loops: 独立回路列表
            G: 拓扑图 (NetworkX)
            opt_config: 包含 num_restarts, tolerance, max_nodes, ee_node, enable_consistency 等配置的字典
            weights: 包含各 Loss 权重的字典
            target_data: 包含 target_twists, target_masks, num_dof 等任务数据的字典

        Returns:
            best_x, best_q, best_metric, initial_metric
        """

        # 1. 解包配置
        NUM_RESTARTS = opt_config['num_restarts']
        max_nodes = self.config['data']['max_nodes']

        # 解包开关
        enable_closure = opt_config['enable_closure']
        enable_mobility = opt_config['enable_mobility']
        enable_task = opt_config['enable_task']
        enable_bennett = opt_config['enable_bennett']

        # [新增] 获取一致性开关 (默认关闭以防旧配置报错)
        enable_consistency = opt_config.get('enable_consistency', False)

        # 记录全局最优 (Across Restarts)
        best_metric_global = float('inf')
        best_x_global = None
        best_q_global = None
        initial_metric_log = 0.0

        # [新增] 预计算末端路径 (用于一致性 Loss)
        # 一致性检查需要知道从 Base 到 EE 的路径
        path_to_ee = []
        if enable_consistency:
            try:
                graph_nodes = list(G.nodes())
                if graph_nodes:
                    base_node = min(graph_nodes)
                    # 获取配置的 EE 节点，如果不存在则使用最大节点索引
                    config_ee = opt_config.get('ee_node', -1)
                    if config_ee in graph_nodes:
                        final_ee = config_ee
                    else:
                        final_ee = max(graph_nodes)

                    path_to_ee = nx.shortest_path(G, source=base_node, target=final_ee)
            except Exception as e:
                # 如果找不到路径（例如图不连通），则无法计算一致性
                path_to_ee = []

        # 2. 多重启动循环
        for attempt in range(NUM_RESTARTS):

            # --- 2.1 初始化变量 ---
            # x_opt: 从 DiT 初值克隆 (保证起点一致)
            x_opt = x_init_raw.clone().requires_grad_(True)
            # q_opt: 全局均匀随机 [-pi, pi]
            q_opt = torch.empty(max_nodes, device=self.device).uniform_(-math.pi, math.pi).requires_grad_(True)

            # 强制初始化合法 (防退化)
            with torch.no_grad():
                x_opt.clamp_(-1.0, 1.0)
                # 强制杆长 a (通道2) > -0.95 (物理值 > 0.5)
                x_opt[:, 2, :, :] = torch.clamp(x_opt[:, 2, :, :], min=-0.95)

            optimizer = optim.Adam([
                {'params': x_opt, 'lr': 0.005},
                {'params': q_opt, 'lr': 0.05}
            ])

            # 单次 attempt 的最优记录
            best_metric_local = float('inf')
            best_x_local = x_opt.detach().clone()
            best_q_local = q_opt.detach().clone()

            # --- 2.2 梯度下降循环 (100步) ---
            for step in range(100):
                optimizer.zero_grad()

                # (a) 正向计算
                x_phys = self.diffusion_model.apply_physics_constraints(x_opt, clamp=True)
                structure = x_phys[0].permute(1, 2, 0)

                current_metric = 0.0
                loss = 0.0

                # (b) Loss 计算

                # 1. 闭环 Loss
                if enable_closure:
                    loss_c = compute_loop_errors(structure, q_opt, loops)
                    loss += loss_c
                    current_metric += loss_c.item()

                # 2. Bennett Loss
                if enable_bennett:
                    loss_b = compute_bennett_geometry_error(structure, loops)
                    loss += weights['bennett'] * loss_b
                    current_metric += loss_b.item()

                # 3. 可动性 Loss (Eigen-Loss)
                loss_m = torch.tensor(0.0, device=self.device)
                if enable_mobility:
                    loss_m = compute_mobility_loss_eigen(
                        structure, q_opt, loops, num_dof=target_data['num_dof'],
                        gap_threshold=opt_config['gap_threshold']
                    )
                    loss += weights['mobility'] * loss_m
                    current_metric += loss_m.item()

                # 4. 任务 Loss (Eigen-Loss)
                loss_t = torch.tensor(0.0, device=self.device)
                if enable_task:
                    ee_idx = opt_config['ee_node']
                    loss_t = compute_task_loss_eigen(
                        structure, q_opt, loops, G, ee_idx,
                        target_data['twists'], target_data['masks']
                    )
                    loss += weights['task'] * loss_t
                    current_metric += loss_t.item()

                # [修改] 5. 全周一致性 Loss (Consistency Loss)
                loss_cons = torch.tensor(0.0, device=self.device)
                if enable_consistency and path_to_ee:
                    # 直接使用 target_data 中的完整数据
                    # 如果没有任务数据，传入 None，函数内部会自动处理
                    tgt_twists = None
                    tgt_masks = None

                    if enable_task and target_data.get('twists') is not None:
                        tgt_twists = target_data['twists']  # (K, 6)
                        tgt_masks = target_data['masks']  # (K, 6)

                    loss_cons = compute_motion_consistency_loss(
                        structure, q_opt, loops, path_to_ee,
                        target_twists=tgt_twists, target_masks=tgt_masks  # <--- 传入完整列表
                    )

                    w_cons = weights.get('consistency', 2.0)
                    loss += w_cons * loss_cons
                    current_metric += loss_cons.item()

                # (c) Snapshot (在 Backward 之前记录)
                if step == 0 and attempt == 0: initial_metric_log = current_metric

                if current_metric < best_metric_local:
                    best_metric_local = current_metric
                    best_x_local = x_opt.detach().clone()
                    best_q_local = q_opt.detach().clone()

                # (d) 辅助 Loss (边界 + 正则)
                _, _, a, alpha, offset = torch.chunk(x_phys, 5, dim=1)
                loss_bounds = torch.relu(-a).sum() + \
                              torch.relu(0.5 - a).sum() * 50.0 + \
                              torch.relu(torch.abs(offset) - 25.0).sum()
                loss += 10.0 * loss_bounds

                # (e) 更新
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x_opt, q_opt], max_norm=1.0)
                optimizer.step()

                # (f) 投影 (Projected Gradient)
                with torch.no_grad():
                    x_opt.data.clamp_(-1.0, 1.0)
                    x_opt.data[:, 2, :, :].clamp_(min=-0.95)
                    q_opt.data.clamp_(-math.pi, math.pi)

                # (g) 调试打印 (仅打印第一个样本的第一次尝试)
                if mech_idx == 0 and attempt == 0 and (step == 0 or (step + 1) % 20 == 0):
                    # 动态生成日志字符串，如果有新 Loss 则显示
                    log_str = (f"  [Debug Mech 0] Step {step + 1:03d}/100 | "
                               f"Total: {loss.item():.2f} | Metric: {current_metric:.4f} | "
                               f"M: {loss_m.item():.4f}")
                    if enable_task: log_str += f" | T: {loss_t.item():.4f}"
                    if enable_consistency: log_str += f" | C: {loss_cons.item():.4f}"
                    print(log_str)

            # --- 2.3 更新全局最优 ---
            if best_metric_local < best_metric_global:
                best_metric_global = best_metric_local
                best_x_global = best_x_local
                best_q_global = best_q_local
                if best_metric_global < 1e-4: break

        return best_x_global, best_q_global, best_metric_global, initial_metric_log

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