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
from ..solver.simple_kinematics import compute_loop_errors, compute_bennett_geometry_error


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

    # --- (核心修改!) 指标5: Bennett 检查 ---
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

    # 2. 修改 _check_kinematic_feasibility 利用传入的参数
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
