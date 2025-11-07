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

        # --- 加载新的指标配置 ---
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
        """辅助函数, 打印所有启用的评估指标及其权重。"""
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
        """
        计算总奖励 R_total.
        动态调用“通用”指标和“特定标签”指标.
        """
        G = self._build_graph(mechanism_tensor)

        if G.number_of_nodes() == 0:
            return -1.0

        total_reward = 0.0

        # --- 1. 运行“通用”指标 ---
        for name, conf in self.common_indicator_config.items():
            if not conf.get('enable', False):
                continue

            # (调用指标函数的逻辑不变)
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
            if score <= -0.9:  # 快速失败
                return total_reward

        # --- 2. 运行“特定标签”指标 ---
        if target_label and target_label in self.label_specific_indicator_config:
            specific_indicators = self.label_specific_indicator_config[target_label]
            for name, conf in specific_indicators.items():
                if not conf.get('enable', False):
                    continue

                # (调用指标函数的逻辑不变)
                try:
                    indicator_function = getattr(self, f"_{name}")
                except AttributeError:
                    # (我们在这里假设 _check_grashof_condition 尚未实现)
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
                if score <= -0.9:  # 快速失败
                    return total_reward

        return total_reward

    # --- 2. 图形构建 (Helper) ---
    def _build_graph(self, tensor):
        """
        从 (N, N, 4) 张量构建 NetworkX 图.
        使用 'exists' 通道判断连接 (双向确认).
        返回最大连通组件.
        """
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

        if not nodes_with_edges:
            return nx.Graph()

        G_sub = G.subgraph(nodes_with_edges).copy()
        if G_sub.number_of_nodes() == 0:
            return nx.Graph()

        try:
            largest_cc_nodes = max(nx.connected_components(G_sub), key=len)
            G_main = G_sub.subgraph(largest_cc_nodes).copy()
        except ValueError:
            return nx.Graph()

        return G_main

    # --- 3. 模块化指标函数 (Indicators) ---

    def _get_error_score(self, error_value, threshold):
        """
        (新!) 计算归一化的奖励/惩罚分数, 范围 [-1, 1].
        - error = 0       -> score = 1.0 (满分奖励)
        - error = threshold -> score = 0.0 (中立)
        - error = 2*threshold -> score = -1.0 (满分惩罚)
        """
        if threshold < 1e-6:  # 避免除零
            return 1.0 if error_value < 1e-6 else -1.0

        score = (threshold - error_value) / threshold
        return max(0.0, min(1.0, score))  # 钳位在 [-1, 1]

    def _calculate_DoF(self, G, tensor):
        """(辅助函数) 计算平面机构的自由度。"""
        num_links = G.number_of_edges()
        num_joints = G.number_of_nodes()
        if num_links < 3 or num_joints < 3:
            return -99
            # M = 3 * (L-1) - 2*J (L=边=links, J=点=joints)
        dof = 3 * (num_links - 1) - 2 * num_joints
        return dof

    def _check_connectivity(self, G, tensor, **kwargs):
        """指标1: 检查基本连通性和复杂性 (硬约束)."""
        if G.number_of_nodes() < 4:
            return -1.0  # 失败 (硬惩罚)
        return 1.0  # 通过

    def _check_dof(self, G, tensor, **kwargs):
        """指标2: 检查自由度是否为 1 (软约束)."""
        dof = self._calculate_DoF(G, tensor)
        if dof == 1:
            return 1.0  # 奖励 (DoF=1)
        elif dof == -99:
            return -1.0  # 硬惩罚无效图
        else:
            return -0.5  # 惩罚 (卡死或欠约束)

    def _check_topology_similarity(self, G, tensor, **kwargs):
        """指标3: 检查拓扑相似性 (偏好, 奖励加成)."""
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if num_nodes == 4 and num_edges == 4:
            degrees = [d for n, d in G.degree()]
            if all(d == 2 for d in degrees):
                return 1.0  # 是4杆环
        return 0.0  # 中立

    def _check_node_count_penalty(self, G, tensor, conf, **kwargs):
        """指标4: 节点数量惩罚 (偏好塑造)."""
        # --- 从传入的 conf 中读取参数 ---
        target_nodes = conf.get('target_nodes', 4)
        node_penalty_scale = conf.get('node_penalty_scale', 0.05)

        num_nodes = G.number_of_nodes()
        deviation = abs(num_nodes - target_nodes)
        penalty = - node_penalty_scale * (deviation ** 2)
        return max(-1.0, penalty)

    # --- 指标5: Bennett 检查 (严格的序贯/门控奖励) ---
    def _check_is_bennett(self, G, tensor, conf, current_cycle=None, total_cycles=None, **kwargs):  # <-- (核心修改!)
        """
        (已修正!) 指标5: 检查机构与 Bennett 约束的接近程度 (严格序贯/门控奖励).
        (新!) 增加了基于 current_cycle 的课程学习门控.
        """

        # --- 从传入的 conf 中读取参数 ---
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
            # --- 计算训练进度 ---
            progress_percent = 1.0  # 默认为 100% (运行所有检查)
            if current_cycle is not None and total_cycles is not None and total_cycles > 0:
                # +1 使 cycle 0 成为 "第 1 周期"
                progress_percent = (current_cycle + 1) / total_cycles

            nodes = list(G.nodes())
            path_edges = list(nx.find_cycle(G, source=nodes[0]))
            if len(path_edges) != 4: return partial_score

            print('\n')
            print(path_edges)

            # --- 门槛 2: 检查 d=0 (混合门控) ---
            D_REWARD_TOTAL = 0.3
            D_REWARD_PER_PARAM = D_REWARD_TOTAL / 8.0
            all_d_are_zero = True

            ordered_params = []

            for u, v in path_edges:
                d_u = tensor[u, v, 3]
                d_v = tensor[v, u, 3]
                print("d_u", d_u, "d_v", d_v)

                if abs(d_u) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False

                if abs(d_v) <= existence_threshold:
                    partial_score += D_REWARD_PER_PARAM
                else:
                    all_d_are_zero = False

                a = tensor[u, v, 1];
                alpha = tensor[u, v, 2]
                if a < 0.0 or alpha < 0.0: return partial_score
                ordered_params.append({'a': a, 'alpha': alpha})

            if len(ordered_params) != 4: return partial_score

            if not all_d_are_zero:
                print(f"d-value check FAILED GATE. Final score: {partial_score}")
                return partial_score

            print(f"d-value check PASSED GATE. Current score: {partial_score}")

            p1, p2, p3, p4 = ordered_params

            # --- 门槛 3: 对边相等 (奖励/惩罚) ---
            A_REWARD_MAX = 0.2
            ALPHA_REWARD_MAX = 0.2

            a_error_1 = abs(p1['a'] - p3['a']) / (max(p1['a'], p3['a']) + 1e-6)
            a_error_2 = abs(p2['a'] - p4['a']) / (max(p2['a'], p4['a']) + 1e-6)
            alpha_error_1 = abs(p1['alpha'] - p3['alpha']) / (max(p1['alpha'], p3['alpha']) + 1e-6)
            alpha_error_2 = abs(p2['alpha'] - p4['alpha']) / (max(p2['alpha'], p4['alpha']) + 1e-6)

            print("a_error_1", a_error_1, "a_error_2", a_error_2)
            print("alpha_error_1", alpha_error_1, "alpha_error_2", alpha_error_2)

            all_sides_equal = (a_error_1 < bennett_error_threshold and
                               a_error_2 < bennett_error_threshold and
                               alpha_error_1 < bennett_error_threshold and
                               alpha_error_2 < bennett_error_threshold)

            if not all_sides_equal:
                if a_error_1 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if a_error_2 < bennett_error_threshold: partial_score += (A_REWARD_MAX / 2.0)
                if alpha_error_1 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                if alpha_error_2 < bennett_error_threshold: partial_score += (ALPHA_REWARD_MAX / 2.0)
                print(f"Side equality check FAILED GATE. Final score: {partial_score}")
                return partial_score

            partial_score += A_REWARD_MAX + ALPHA_REWARD_MAX
            print(f"Side-equality check PASSED GATE. Current score: {partial_score}")

            # --- 门槛 4: Bennett 正弦法则 (0.9 -> 1.0) ---
            print("Proceeding to Bennett Sine Law check (Final Stage)...")
            BENNETT_REWARD_MAX = 0.1
            a1 = p1['a']
            alpha1 = p1['alpha']
            a2 = p2['a']
            alpha2 = p2['alpha']

            if alpha1 > existence_threshold and alpha2 > existence_threshold and \
                    alpha1 < math.pi - existence_threshold and alpha2 < math.pi - existence_threshold:

                sin_alpha1 = math.sin(alpha1)
                sin_alpha2 = math.sin(alpha2)

                val1 = a1 / sin_alpha2
                val2 = a2 / sin_alpha1
                bennett_error_rel = abs(val1 - val2) / (max(abs(val1), abs(val2)) + 1e-6)
                print("bennett_error_rel", bennett_error_rel)

                if bennett_error_rel < bennett_error_threshold:
                    partial_score += BENNETT_REWARD_MAX

            print("final_score:", partial_score)

            return partial_score

        except Exception as e:
            # print(f"Error during Bennett check: {e}") # 调试
            return partial_score