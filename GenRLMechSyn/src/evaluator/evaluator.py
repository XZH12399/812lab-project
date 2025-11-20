# src/evaluator/evaluator.py
import numpy as np
import networkx as nx
import sys
import math


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

    def evaluate(self, mechanism_tensor, target_label=None, current_cycle=None, total_cycles=None):
        G = self._build_graph(mechanism_tensor)
        if G.number_of_nodes() == 0: return -1.0
        total_reward = 0.0
        for name, conf in self.common_indicator_config.items():
            if not conf.get('enable', False): continue
            try:
                indicator_function = getattr(self, f"_{name}")
            except AttributeError:
                print(f"警告: 配置文件 中的 '{name}' 无法在 evaluator.py 中找到 '_{name}' 函数。")
                continue
            try:
                score = indicator_function(
                    G, mechanism_tensor,
                    conf=conf,
                    current_cycle=current_cycle, total_cycles=total_cycles
                )
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
                    score = indicator_function(
                        G, mechanism_tensor,
                        conf=conf,
                        current_cycle=current_cycle, total_cycles=total_cycles
                    )
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
    def _check_is_bennett(self, G, tensor, conf, current_cycle=None, total_cycles=None, **kwargs):
        """
        (已修改!) joint_type (通道 1) 检查逻辑被简化。
        通道: [0:exists, 1:joint_type, 2:a, 3:alpha, 4:offset]
        """
        existence_threshold = conf.get('existence_threshold', 0.05)
        bennett_error_threshold = conf.get('bennett_error_threshold', 0.05)

        # --- 门槛 1: 拓扑 ---
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if not (num_nodes == 4 and num_edges == 4): return 0.0
        degrees = [d for n, d in G.degree()]
        if not all(d == 2 for d in degrees): return 0.0

        partial_score = 0.2  # 基础分: 这是一个4杆环

        try:
            # --- (核心修改!) 门槛 1b: 检查是否所有关节都是 R 副 (+1) ---
            nodes = list(G.nodes())
            for node_i in nodes:
                # 新逻辑: 只需检查一个元素 (e.g., [i, 0] 或 [i, i])
                # 因为 sample() 和 dataloader 保证了整行 [i, :, 1] 都是一样的

                # 我们从 tensor[node_i, 0, 1] 读取 (0 是任意选择的列索引)
                joint_type_i = tensor[node_i, 0, 1]  # 通道 1: joint_type

                if joint_type_i <= 0:  # 如果是 -1 (P 副) 或 0 (无效/未连接)
                    return 0.0  # 硬失败, Bennett 必须是全 R 副
            # --- 修改结束 ---

            # --- 训练进度 ---
            progress_percent = 1.0
            if current_cycle is not None and total_cycles is not None and total_cycles > 0:
                progress_percent = (current_cycle + 1) / total_cycles

            path_edges = list(nx.find_cycle(G, source=nodes[0]))
            if len(path_edges) != 4: return partial_score

            # --- 门槛 2: 检查 offset=0 (d=0) ---
            D_REWARD_TOTAL = 0.3
            D_REWARD_PER_PARAM = D_REWARD_TOTAL / 8.0
            all_d_are_zero = True
            ordered_params = []

            for u, v in path_edges:
                d_u = tensor[u, v, 4]  # 通道 4: offset
                d_v = tensor[v, u, 4]
                if abs(d_u) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False
                if abs(d_v) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False

                a = tensor[u, v, 2];  # 通道 2: a
                alpha = tensor[u, v, 3]  # 通道 3: alpha
                if a < 0.0 or alpha < 0.0: return partial_score
                ordered_params.append({'a': a, 'alpha': alpha})

            if len(ordered_params) != 4: return partial_score
            if not all_d_are_zero:
                return partial_score

            p1, p2, p3, p4 = ordered_params

            # --- 门槛 3: 对边相等 ---
            A_REWARD_MAX = 0.2
            ALPHA_REWARD_MAX = 0.2
            a_error_1 = abs(p1['a'] - p3['a']) / (max(p1['a'], p3['a']) + 1e-6)
            a_error_2 = abs(p2['a'] - p4['a']) / (max(p2['a'], p4['a']) + 1e-6)
            alpha_error_1 = abs(p1['alpha'] - p3['alpha']) / (max(p1['alpha'], p3['alpha']) + 1e-6)
            alpha_error_2 = abs(p2['alpha'] - p4['alpha']) / (max(p2['alpha'], p4['alpha']) + 1e-6)
            all_sides_equal = (a_error_1 < bennett_error_threshold and
                               a_error_2 < bennett_error_threshold and
                               alpha_error_1 < bennett_error_threshold and
                               alpha_error_2 < bennett_error_threshold)
            if not all_sides_equal:
                if a_error_1 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if a_error_2 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if alpha_error_1 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                if alpha_error_2 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                return partial_score
            partial_score += A_REWARD_MAX + ALPHA_REWARD_MAX

            # --- 门槛 4: Bennett 正弦法则 ---
            BENNETT_REWARD_MAX = 0.1
            a1, alpha1 = p1['a'], p1['alpha']
            a2, alpha2 = p2['a'], p2['alpha']
            if alpha1 > existence_threshold and alpha2 > existence_threshold and \
                    alpha1 < math.pi - existence_threshold and alpha2 < math.pi - existence_threshold:
                sin_alpha1 = math.sin(alpha1)
                sin_alpha2 = math.sin(alpha2)
                val1 = a1 / sin_alpha2
                val2 = a2 / sin_alpha1
                bennett_error_rel = abs(val1 - val2) / (max(abs(val1), abs(val2)) + 1e-6)
                if bennett_error_rel < bennett_error_threshold:
                    partial_score += BENNETT_REWARD_MAX
            return partial_score
        except Exception as e:
            return partial_score
