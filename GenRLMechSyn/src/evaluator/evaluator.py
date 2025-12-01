# src/evaluator/evaluator.py
import numpy as np
import networkx as nx
import sys
import math
import torch

try:
    from src.solver.kinematics_layer import DifferentiableKinematicsLayer
except ImportError:
    # 防止在未完成配置时报错，给个占位
    DifferentiableKinematicsLayer = None

from ..solver.utils import tensor_to_graph, find_independent_loops
from ..solver.simple_kinematics import (
    compute_loop_errors,
    compute_bennett_geometry_error,
    compute_all_joint_screws, # <--- 确保导入这个
    compute_motion_consistency_loss  # <--- [新增] 添加这个导入
)


class MechanismEvaluator:
    # ( ... __init__, _print_enabled_indicators, evaluate, _build_graph ... 均不变)
    def __init__(self, config):
        self.config = config
        try:
            self.max_nodes = config['data']['max_nodes']
        except KeyError:
            print("[警告] 配置文件中缺少 data.max_nodes, 将使用默认值 30")
            self.max_nodes = 30
        eval_conf = config.get('evaluator_config', {})
        try:
            self.common_indicator_config = eval_conf['common_indicators']
            self.label_specific_indicator_config = eval_conf['label_specific_indicators']
        except KeyError:
            print("[致命错误] 配置文件 中未找到 'common_indicators' 或 'label_specific_indicators' 块。")
            self.common_indicator_config = {}
            self.label_specific_indicator_config = {}
        print("评估模块已初始化 (使用【动态指标】系统)。")
        # --- 初始化求解器 ---
        self.device = torch.device(config['training'].get('device', 'cpu'))
        if DifferentiableKinematicsLayer:
            self.solver = DifferentiableKinematicsLayer(config).to(self.device)
            print("求解器 (Kinematics Solver) 已加载。")
        else:
            self.solver = None
            print("[警告] 未能加载 DifferentiableKinematicsLayer。")
        self._print_enabled_indicators()

    def _print_enabled_indicators(self):
        print("--- 启用的“通用”指标 ---")
        if not self.common_indicator_config:
            print("  (无通用指标)")
        for name, conf in self.common_indicator_config.items():
            if conf.get('enable', False):
                print(f"  [ON]  {name} (Weight: {conf.get('weight', 0.0)})")
        print("--- 启用的“特定标签”指标 ---")
        if not self.label_specific_indicator_config:
            print("  (无特定指标)")
        for label_name, indicators in self.label_specific_indicator_config.items():
            print(f"  (标签: '{label_name}')")
            for name, conf in indicators.items():
                if conf.get('enable', False):
                    print(f"    [ON]  {name} (Weight: {conf.get('weight', 0.0)})")
        print("------------------------")

    # 1. 修改 evaluate 函数签名，增加 optimized_joint_angles 参数
    def evaluate(self, mechanism_tensor, target_label=None,
                 current_cycle=None, total_cycles=None,
                 optimized_joint_angles=None,
                 known_loops=None):  # <--- 新增参数

        G = self._build_graph(mechanism_tensor)
        if G.number_of_nodes() == 0: return -1.0

        total_reward = 0.0

        # 将参数打包，方便传递给各个指标函数
        context = {
            'current_cycle': current_cycle,
            'total_cycles': total_cycles,
            'optimized_joint_angles': optimized_joint_angles,
            'known_loops': known_loops
        }
        for name, conf in self.common_indicator_config.items():
            if not conf.get('enable', False): continue
            try:
                indicator_function = getattr(self, f"_{name}")
            except AttributeError:
                print(f"警告: 配置文件 中的 '{name}' 无法在 evaluator.py 中找到 '_{name}' 函数。")
                continue
            try:
                score = indicator_function(G, mechanism_tensor, conf=conf, **context)  # <--- 传入 context
            except Exception as e:
                print(f"警告: 调用指标函数 '_{name}' 时出错: {e}")
                score = 0.0
            total_reward += conf.get('weight', 0.0) * score
            if score <= -0.9: return total_reward
        if target_label and target_label in self.label_specific_indicator_config:
            specific_indicators = self.label_specific_indicator_config[target_label]
            for name, conf in specific_indicators.items():
                if not conf.get('enable', False): continue
                try:
                    indicator_function = getattr(self, f"_{name}")
                except AttributeError:
                    print(f"警告: 配置文件 中的 '{name}' 无法在 evaluator.py 中找到 '_{name}' 函数。")
                    continue
                try:
                    score = indicator_function(G, mechanism_tensor, conf=conf, **context)  # <--- 传入 context
                except Exception as e:
                    print(f"警告: 调用指标函数 '_{name}' 时出错: {e}")
                    score = 0.0
                total_reward += conf.get('weight', 0.0) * score
                if score <= -0.9: return total_reward
        return total_reward

    def _build_graph(self, tensor):
        exists_matrix = tensor[:, :, 0]
        current_max_nodes = self.max_nodes
        G = nx.Graph()
        nodes_with_edges = set()
        for i in range(current_max_nodes):
            for k in range(i + 1, current_max_nodes):
                if exists_matrix[i, k] > 0.5 and exists_matrix[k, i] > 0.5:
                    G.add_edge(i, k)
                    nodes_with_edges.add(i)
                    nodes_with_edges.add(k)
        if not nodes_with_edges: return nx.Graph()
        G_sub = G.subgraph(nodes_with_edges).copy()
        if G_sub.number_of_nodes() == 0: return nx.Graph()
        try:
            largest_cc_nodes = max(nx.connected_components(G_sub), key=len)
            G_main = G_sub.subgraph(largest_cc_nodes).copy()
        except ValueError:
            return nx.Graph()
        return G_main

    # ( ... _get_error_score, _calculate_DoF, _check_connectivity, _check_dof,
    #   _check_topology_similarity, _check_node_count_penalty ... 均不变)
    def _get_error_score(self, error_value, threshold):
        if threshold < 1e-6:
            return 1.0 if error_value < 1e-6 else -1.0
        score = (threshold - error_value) / threshold
        return max(0.0, min(1.0, score))

    def _calculate_DoF(self, G, tensor):
        num_links = G.number_of_edges()
        num_joints = G.number_of_nodes()
        if num_links < 3 or num_joints < 3: return -99
        dof = 3 * (num_links - 1) - 2 * num_joints
        return dof

    def _check_connectivity(self, G, tensor, **kwargs):
        if G.number_of_nodes() < 4: return -1.0
        return 1.0

    def _check_dof(self, G, tensor, **kwargs):
        dof = self._calculate_DoF(G, tensor)
        if dof == 1:
            return 1.0
        elif dof == -99:
            return -1.0
        else:
            return -0.5

    def _check_topology_similarity(self, G, tensor, **kwargs):
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if num_nodes == 4 and num_edges == 4:
            degrees = [d for n, d in G.degree()]
            if all(d == 2 for d in degrees):
                return 1.0
        return 0.0

    def _check_node_count_penalty(self, G, tensor, conf, **kwargs):
        target_nodes = conf.get('target_nodes', 4)
        node_penalty_scale = conf.get('node_penalty_scale', 0.05)
        num_nodes = G.number_of_nodes()
        deviation = abs(num_nodes - target_nodes)
        penalty = - node_penalty_scale * (deviation ** 2)
        return max(-1.0, penalty)

    def _check_is_bennett(self, G, tensor, conf, known_loops=None, **kwargs):
        """
        [修改版] 复用 simple_kinematics 中的计算逻辑
        """
        # --- 1. 门控逻辑 (Evaluator 独有) ---
        # 只有在拓扑满足要求时才计算
        if known_loops is None:
            loops = find_independent_loops(G)
        else:
            loops = known_loops

        if not loops or len(loops[0]) != 4:
            return 0.0  # 门控拦截：不是4杆，直接0分，不计算后续

        # --- 2. 准备数据 ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        structure_tensor = torch.tensor(tensor, dtype=torch.float32, device=device)

        # --- 3. 调用公共计算函数 ---
        # 这里的计算逻辑和 Optimizer 完全一样
        geo_error_tensor = compute_bennett_geometry_error(structure_tensor, loops)
        geo_error = geo_error_tensor.item()

        # --- 4. 评分逻辑 (Evaluator 独有) ---
        print(f"    [Evaluator] Bennett几何误差: {geo_error:.6f}")

        THRESHOLD = 0.1  # 设定一个允许的几何误差阈值

        if geo_error > THRESHOLD:
            return 0.0  # 误差太大，不合格

        # 转化为分数
        return 1.0 * np.exp(-2.0 * geo_error)

    def _check_kinematic_feasibility(self, G, tensor, conf, optimized_joint_angles=None, known_loops=None, **kwargs):
        """
        检查运动学可行性。
        如果提供了 known_loops，直接使用它，确保与优化时的路径一致。
        """
        # =======================================================
        # 模式 A: 极速模式 (使用 Pipeline 传入的优化结果)
        # =======================================================
        if optimized_joint_angles is not None:
            try:
                device = optimized_joint_angles.device
                structure_tensor = torch.tensor(tensor, dtype=torch.float32, device=device)

                # [核心修改] 优先使用传入的 loops
                if known_loops is not None:
                    loops = known_loops
                else:
                    # 只有没传的时候才自己找 (可能方向不一致)
                    loops = find_independent_loops(G)

                if not loops:
                    return -1.0

                # 3. 复用 simple_kinematics 计算误差
                # 这保证了 Evaluator 看到的误差和 Pipeline 优化时的误差逻辑一致
                closure_error = compute_loop_errors(structure_tensor, optimized_joint_angles, loops)

                # (兼容性处理) 防止 compute_loop_errors 返回元组 (loss, mobility_loss)
                if isinstance(closure_error, tuple):
                    closure_error = closure_error[0]

                final_err = closure_error.item()

                # --- [调试信息] 打印真实的误差值 ---
                # 这能让你看到 Evaluator 到底算出了多少误差
                print(f"    [Evaluator] 闭环误差检查: {final_err:.6f}")

                # 4. [关键] 阈值判定
                # 建议放宽到 2.0，容忍一些因为截断(Clamp)带来的微小几何变动。
                # 只要误差在这个范围内，说明机构是基本闭环的。
                THRESHOLD = 0.1

                if final_err > THRESHOLD:
                    print(f"    [Evaluator] 失败: 误差 {final_err:.6f} > {THRESHOLD}")
                    return -1.0

                # 5. 计算得分 (0.0 ~ 1.0)
                # 使用指数函数将误差映射为分数
                return 1.0 * np.exp(-2.0 * final_err)

            except Exception as e:
                print(f"[警告] Evaluator 使用优化参数检查失败: {e}")
                # 如果出错，不要直接返回 -1，尝试回退到模式 B
                pass

        # =======================================================
        # 模式 B: 回退模式 (使用 Solver 从零求解)
        # =======================================================
        # 仅当 optimized_joint_angles 未提供或出错时使用

        if self.solver is None:
            return 0.0

        # 准备数据: Solver 期望 (B, C, H, W) 格式
        tensor_torch = torch.tensor(tensor, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            # 这里的 solver 是 Theseus Layer (如果已加载)
            solved_q, success, error = self.solver(tensor_torch)

        final_err = error.item()

        # 打印调试信息
        print(f"    [Evaluator] Solver 求解误差: {final_err:.6f}")

        if final_err > 2.0:  # 同样使用放宽的阈值
            return -1.0

        return 1.0 * np.exp(-2.0 * final_err)

    def _check_mobility(self, G, tensor, conf, optimized_joint_angles=None, known_loops=None, **kwargs):
        """
        [智能推断版] 检查机构可动性。
        自动根据 target_motion_patterns 的数量推断目标自由度 K。
        """
        if optimized_joint_angles is None: return 0.0
        if known_loops is None:
            loops = find_independent_loops(G)
        else:
            loops = known_loops
        if not loops: return 0.0

        device = optimized_joint_angles.device
        structure = torch.tensor(tensor, dtype=torch.float32, device=device)

        # ==========================================================
        # [核心修改] 自动推断目标自由度 (Target DOF)
        # ==========================================================
        # 1. 尝试从 check_task_performance 中获取模式列表
        task_conf = self.config.get('evaluator_config', {}).get('common_indicators', {}).get('check_task_performance',
                                                                                             {})
        patterns = task_conf.get('target_motion_patterns', [])

        if patterns and len(patterns) > 0:
            # 如果定义了任务模式，自由度必须匹配模式数量 (例如 2T -> 2)
            target_dof = len(patterns)
        else:
            # 2. 如果没定义任务(例如 Bennett)，回退到显式配置或默认值 1
            target_dof = self.config['generation'].get('target_dof', 1)

        # ==========================================================

        try:
            # 1. 计算螺旋
            base_node = min(G.nodes()) if G.number_of_nodes() > 0 else 0
            all_screws, _ = compute_all_joint_screws(structure, optimized_joint_angles, base_node=base_node)

            # 2. 构建雅可比矩阵 J
            constraint_rows = []
            for loop_nodes in loops:
                J_loop = torch.zeros((6, structure.shape[0]), device=device)
                for node_idx in loop_nodes:
                    J_loop[:, node_idx] = all_screws[node_idx]
                constraint_rows.append(J_loop)

            J_global = torch.cat(constraint_rows, dim=0)

            # 3. 构建 Gram 矩阵
            G_mat = J_global.T @ J_global

            # 4. 特征值分解
            eigenvalues = torch.linalg.eigvalsh(G_mat)

            # 5. 检查第 K 个特征值 (索引 K-1)
            check_index = target_dof - 1
            if check_index >= len(eigenvalues):
                check_index = len(eigenvalues) - 1

            critical_eig = eigenvalues[check_index]

            # 打印前几个特征值供观察
            display_count = min(len(eigenvalues), target_dof + 2)
            eigs_str = ", ".join([f"{e:.6f}" for e in eigenvalues[:display_count]])
            print(f"    [Evaluator] Mobility Eigenvalues (Target DOF={target_dof}): [{eigs_str}]")

            # 6. 评分
            threshold = conf.get('threshold', 1e-3)

            if critical_eig.item() > threshold:
                return 0.0  # 自由度不足

            return 1.0 * np.exp(-100.0 * critical_eig.item())

        except Exception as e:
            print(f"    [Evaluator] Mobility Check Error: {e}")
            return 0.0

    def _check_task_performance(self, G, tensor, conf, optimized_joint_angles=None, known_loops=None, **kwargs):
        """
        [统一版] 任务性能检查 (增广 Gram 矩阵法)。
        通过构建 [J_path | Target] 的增广矩阵并计算其最小特征值，
        验证机构是否能在物理上精确执行目标运动模式 (Masked)。
        """
        # 1. 基础依赖检查
        if optimized_joint_angles is None: return 0.0
        if known_loops is None:
            loops = find_independent_loops(G)
        else:
            loops = known_loops
        if not loops: return 0.0

        # 获取目标配置
        target_patterns = conf.get('target_motion_patterns', [])
        if not target_patterns: return 1.0  # 无目标默认通过

        try:
            device = optimized_joint_angles.device
            structure = torch.tensor(tensor, dtype=torch.float32, device=device)

            graph_nodes = list(G.nodes())
            if not graph_nodes: return 0.0

            base_node = min(graph_nodes)

            # 获取配置的 ee_node，如果图中不存在，降级为 max(nodes)
            config_ee = self.config.get('generation', {}).get('ee_node', -1)
            if config_ee in graph_nodes:
                ee_node = config_ee
            else:
                ee_node = max(graph_nodes)

            # 2. 计算所有关节的全局螺旋
            all_screws, _ = compute_all_joint_screws(structure, optimized_joint_angles, base_node=base_node)

            try:
                path_to_ee = nx.shortest_path(G, source=base_node, target=ee_node)
            except:
                return 0.0

            # J_path_full: (6, M) 每一列是路径上一个关节的螺旋
            path_screws_list = [all_screws[u] for u in path_to_ee]
            J_path_full = torch.stack(path_screws_list, dim=1)

            passed_count = 0

            # 4. 逐个模式检查 (应用 Mask)
            for idx, pattern in enumerate(target_patterns):
                # pattern: [wx, wy, wz, vx, vy, vz] (含 None)

                # A. 解析 Mask 和 Target
                valid_indices = []  # 记录哪些维度是有效的 (非 None)
                target_vals = []

                for dim, val in enumerate(pattern):
                    if val is not None:
                        valid_indices.append(dim)
                        target_vals.append(float(val))

                # 如果全为 None，跳过
                if not valid_indices:
                    passed_count += 1
                    continue

                # B. 构建子问题 (Sub-problem)
                # J_sub: (D_valid, M) 只取关注的行
                J_sub = J_path_full[valid_indices, :]

                # Target_sub: (D_valid, 1)
                target_sub = torch.tensor(target_vals, device=device, dtype=torch.float32).unsqueeze(1)

                # C. 构建增广矩阵 [J_sub | Target_sub]
                # 核心逻辑：如果 Target 在 J_sub 的列空间内（任务可达），
                # 那么增广矩阵的列向量组必然线性相关。
                J_aug = torch.cat([J_sub, target_sub], dim=1)

                # D. 计算 Gram 矩阵 G = J_aug.T @ J_aug
                G_aug = J_aug.T @ J_aug

                # E. 特征值分解 (eigvalsh 针对对称矩阵，数值极稳定)
                eigenvalues = torch.linalg.eigvalsh(G_aug)
                min_eig = eigenvalues[0]  # 升序排列，取最小

                print(f"    [Evaluator] 目标 {idx + 1} 增广特征值: {min_eig.item():.8f}")

                # F. 阈值判定
                # 理论上应为 0。考虑到浮点误差，设定一个较小的阈值。
                # 注意：这是奇异值的平方，所以比 SVD 的阈值要更敏感。
                if min_eig.item() < 1e-4:
                    passed_count += 1

            # 5. 评分
            # 必须所有目标模式都满足才算通过 (2T 需要同时满足两个移动)
            if passed_count >= len(target_patterns):
                return 1.0
            else:
                return 0.0

        except Exception as e:
            print(f"    [Evaluator Task Check Error] {e}")
            return 0.0

    def _check_global_consistency(self, G, tensor, conf, optimized_joint_angles=None, known_loops=None, **kwargs):
        """
        [新增] 运动全周一致性检查 (通用指标)。
        基于二阶微分运动学，检查末端加速度是否漂移出速度定义的流形。
        支持多任务模式：会检查所有定义的 Target Patterns 是否都保持了一致性。
        """
        # 1. 前置检查：必须有优化好的关节角
        if optimized_joint_angles is None:
            return 0.0

        # 2. 拓扑准备
        if known_loops is None:
            loops = find_independent_loops(G)
        else:
            loops = known_loops

        if not loops: return 0.0

        graph_nodes = list(G.nodes())
        if not graph_nodes: return 0.0

        # 确定基座和末端路径
        base_node = min(graph_nodes)

        # 尝试获取配置的 EE 节点
        config_ee = self.config.get('generation', {}).get('ee_node', -1)
        if config_ee in graph_nodes:
            ee_node = config_ee
        else:
            ee_node = max(graph_nodes)

        try:
            path_to_ee = nx.shortest_path(G, source=base_node, target=ee_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

        # 3. 准备数据
        device = optimized_joint_angles.device
        structure = torch.tensor(tensor, dtype=torch.float32, device=device)

        try:
            # 4. 解析任务目标 (Target Motion Patterns)
            # 逻辑：尝试从 'check_task_performance' 配置中读取定义的任务
            target_twists = None
            target_masks = None

            # 从 common_indicator_config 中查找任务配置
            task_conf = self.common_indicator_config.get('check_task_performance', {})
            patterns = task_conf.get('target_motion_patterns', [])

            if patterns:
                t_list = []
                m_list = []
                for pat in patterns:
                    # 每个 pattern 是长度为6的列表 (e.g., [0, 0, 1, None, ...])
                    row_t = []
                    row_m = []
                    for val in pat:
                        if val is not None:
                            row_t.append(float(val))
                            row_m.append(1.0)
                        else:
                            row_t.append(0.0)
                            row_m.append(0.0)
                    t_list.append(row_t)
                    m_list.append(row_m)

                target_twists = torch.tensor(t_list, device=device)  # Shape: (K, 6)
                target_masks = torch.tensor(m_list, device=device)  # Shape: (K, 6)

            # 5. 调用二阶求解器计算 Loss
            # 注意：如果 target_twists 为 None，函数内部会自动退化为无任务的一致性检查
            loss_tensor = compute_motion_consistency_loss(
                structure, optimized_joint_angles, loops, path_to_ee,
                target_twists=target_twists, target_masks=target_masks
            )

            loss_val = loss_tensor.item()

            # 仅在调试时打印，避免刷屏
            # print(f"    [Evaluator] Consistency Drift: {loss_val:.6f}")

            # 6. 评分转换
            # 漂移量越小越好。
            # 阈值判定：如果漂移占比 > 5% (0.05)，则认为运动性质不稳定，直接 0 分
            THRESHOLD = conf.get('threshold', 0.05)

            if loss_val > THRESHOLD:
                return 0.0

                # 使用指数衰减打分，漂移为0时得1分
            # 系数 -10.0 意味着如果漂移是 0.05，得分约为 exp(-0.5) = 0.6
            return 1.0 * np.exp(-10.0 * loss_val)

        except Exception as e:
            # 捕获 SVD 不收敛等潜在数值错误，不中断评估
            print(f"    [Evaluator Consistency Warning] {e}")
            return 0.0
