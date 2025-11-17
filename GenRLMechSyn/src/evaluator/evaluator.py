# src/evaluator/evaluator.py
import numpy as np
import networkx as nx
import sys
import math


class MechanismEvaluator:
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

    # --- 1. 核心评估函数 (Orchestrator) ---
    def evaluate(self, mechanism_tensor, target_label=None, current_cycle=None, total_cycles=None):
        G = self._build_graph(mechanism_tensor)
        if G.number_of_nodes() == 0: return -1.0
        total_reward = 0.0
        # 通用指标循环
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
        # 特定标签指标循环
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

    # --- 2. 图形构建 (Helper) ---
    def _build_graph(self, tensor):
        """
        从 (N, N, 4) 张量构建 NetworkX 图.
        使用 'mixed' 通道 (0) 的 *非对角线* 判断连接 (双向确认).
        """
        mixed_matrix = tensor[:, :, 0]  # 通道 0 是 mixed
        current_max_nodes = self.max_nodes
        G = nx.Graph()
        nodes_with_edges = set()
        for i in range(current_max_nodes):
            for k in range(i + 1, current_max_nodes):
                # 检查非对角线
                if mixed_matrix[i, k] > 0.5 and mixed_matrix[k, i] > 0.5:
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

    # --- 3. 模块化指标函数 (Indicators) ---
    def _get_error_score(self, error_value, threshold):
        if threshold < 1e-6:
            return 1.0 if error_value < 1e-6 else -1.0
        score = (threshold - error_value) / threshold
        return max(0.0, min(1.0, score))  # 钳位在 [-1, 1]

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

    # --- 指标5: Bennett 检查 ---
    def _check_is_bennett(self, G, tensor, conf, current_cycle=None, total_cycles=None, **kwargs):
        """
        检查机构与 Bennett 约束的接近程度.
        通道: [0:mixed, 1:a, 2:alpha, 3:offset]
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
            # --- 门槛 1b: 检查是否所有关节都是 R 副 (1.0) ---
            nodes = list(G.nodes())
            for node_i in nodes:
                # 检查关节 i 的类型.
                # 我们从 tensor[i, i, 0] (对角线) 读取
                joint_type_i = tensor[node_i, node_i, 0]  # 通道 0 (mixed), 对角线

                if joint_type_i <= 0.5:  # 如果是 0 (P 副)
                    return 0.0 # 硬失败, Bennett 必须是全 R 副

            # --- 训练进度 ---
            progress_percent = 1.0
            if current_cycle is not None and total_cycles is not None and total_cycles > 0:
                progress_percent = (current_cycle + 1) / total_cycles

            # nodes = list(G.nodes()) # 已在上面获取
            path_edges = list(nx.find_cycle(G, source=nodes[0]))
            if len(path_edges) != 4: return partial_score

            # print('\n')
            # print(path_edges)

            # --- 门槛 2: 检查 offset=0 (d=0) ---
            D_REWARD_TOTAL = 0.3
            D_REWARD_PER_PARAM = D_REWARD_TOTAL / 8.0
            all_d_are_zero = True
            ordered_params = []

            for u, v in path_edges:
                # d -> offset, 通道 3
                d_u = tensor[u, v, 3]  # 通道 3: offset
                d_v = tensor[v, u, 3]  # 通道 3: offset
                # print("d_u", d_u, "d_v", d_v)

                if abs(d_u) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False
                if abs(d_v) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False

                # a 通道 1, alpha 通道 2
                a = tensor[u, v, 1];  # 通道 1: a
                alpha = tensor[u, v, 2]  # 通道 2: alpha

                if a < 0.0 or alpha < 0.0: return partial_score
                ordered_params.append({'a': a, 'alpha': alpha})

            if len(ordered_params) != 4: return partial_score
            if not all_d_are_zero:
                # print(f"offset-value check FAILED GATE. Final score: {partial_score}")
                return partial_score
            # print(f"offset-value check PASSED GATE. Current score: {partial_score}")

            p1, p2, p3, p4 = ordered_params

            # --- 门槛 3: 对边相等 ---
            A_REWARD_MAX = 0.2
            ALPHA_REWARD_MAX = 0.2
            a_error_1 = abs(p1['a'] - p3['a']) / (max(p1['a'], p3['a']) + 1e-6)
            a_error_2 = abs(p2['a'] - p4['a']) / (max(p2['a'], p4['a']) + 1e-6)
            alpha_error_1 = abs(p1['alpha'] - p3['alpha']) / (max(p1['alpha'], p3['alpha']) + 1e-6)
            alpha_error_2 = abs(p2['alpha'] - p4['alpha']) / (max(p2['alpha'], p4['alpha']) + 1e-6)
            # print("a_error_1", a_error_1, "a_error_2", a_error_2)
            # print("alpha_error_1", alpha_error_1, "alpha_error_2", alpha_error_2)
            all_sides_equal = (a_error_1 < bennett_error_threshold and
                               a_error_2 < bennett_error_threshold and
                               alpha_error_1 < bennett_error_threshold and
                               alpha_error_2 < bennett_error_threshold)
            if not all_sides_equal:
                if a_error_1 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if a_error_2 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if alpha_error_1 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                if alpha_error_2 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                # print(f"Side equality check FAILED GATE. Final score: {partial_score}")
                return partial_score
            partial_score += A_REWARD_MAX + ALPHA_REWARD_MAX
            # print(f"Side-equality check PASSED GATE. Current score: {partial_score}")

            # --- 门槛 4: Bennett 正弦法则 ---
            # print("Proceeding to Bennett Sine Law check (Final Stage)...")
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
                # print("bennett_error_rel", bennett_error_rel)
                if bennett_error_rel < bennett_error_threshold:
                    partial_score += BENNETT_REWARD_MAX
            # print("final_score:", partial_score)
            return partial_score
        except Exception as e:
            # print(f"Error during Bennett check: {e}") # 调试
            return partial_score