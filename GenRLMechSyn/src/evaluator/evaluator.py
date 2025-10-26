# src/evaluator/evaluator.py
import numpy as np
import networkx as nx
import sys


class MechanismEvaluator:
    def __init__(self, config):
        self.config = config
        self.max_nodes = config['data']['max_nodes']
        self.adjacency_threshold = 0.1

        # --- (新!) 加载动态指标配置 ---
        try:
            self.indicator_config = config['evaluator_config']['indicators']
        except KeyError:
            print("[致命错误] 配置文件中未找到 'evaluator_config' 块。")
            sys.exit(1)

        print("评估模块已初始化 (使用【动态指标】系统)。")
        self._print_enabled_indicators()

    def _print_enabled_indicators(self):
        """辅助函数, 打印所有启用的评估指标及其权重。"""
        print("--- 启用的评估指标 ---")
        for name, conf in self.indicator_config.items():
            if conf['enable']:
                print(f"  [ON]  {name} (Weight: {conf['weight']})")
            else:
                print(f"  [OFF] {name}")
        print("------------------------")

    # -----------------------------------------------------------------
    # --- 1. 核心评估函数 (Orchestrator) ---
    # -----------------------------------------------------------------

    def evaluate(self, mechanism_tensor):
        """
        计算总奖励 R_total.
        (动态地调用所有在 config 中启用的指标)
        """
        # 1. 解码张量
        G = self._build_graph(mechanism_tensor)

        # 2. (硬编码) 检查空图
        if G.number_of_nodes() == 0:
            return -1.0  # 垃圾, 直接返回

        total_reward = 0.0

        # 3. 动态循环所有指标
        for name, conf in self.indicator_config.items():
            # 检查开关
            if not conf['enable']:
                continue

            # 1. 获取指标函数 (e.g., self._check_connectivity)
            try:
                indicator_function = getattr(self, f"_{name}")
            except AttributeError:
                print(f"警告: 配置文件中的 '{name}' 无法在 evaluator.py 中找到 '_{name}' 函数。")
                continue

            # 2. 调用函数, 获取分数
            score = indicator_function(G)

            # 3. 累加加权分数
            total_reward += conf['weight'] * score

            # 4. --- (关键!) "Fail-Fast" 机制 ---
            # 如果一个 "硬约束" 函数返回了灾难性的低分 (e.g., -1.0),
            # 我们就没必要继续评估了 (e.g., 不要在非连通图上检查 DoF)。
            # 我们定义 -0.9 为 "失败" 阈值。
            if score <= -0.9:
                return total_reward  # 立即停止并返回惩罚

        return total_reward

    # -----------------------------------------------------------------
    # --- 2. 图形构建 (Helper) ---
    # -----------------------------------------------------------------

    def _build_graph(self, tensor):
        """
        (最终修正版 v2!) 从 (30, 30, 4) 的张量构建 NetworkX 图.
        直接使用 'exists' 通道 (索引 0) 判断连接.
        **要求双向确认**: exists[i, k] 和 exists[k, i] 都必须 > 0.5.
        返回最大连通组件.
        """
        # --- 获取 exists 通道 ---
        exists_matrix = tensor[:, :, 0]

        # 1. 构建初始图 (遍历所有可能的节点对 i < k)
        G = nx.Graph()
        for i in range(self.max_nodes):
            for k in range(i + 1, self.max_nodes):
                # --- 核心判断逻辑 (双向确认) ---
                # 检查 i -> k 和 k -> i 的信号是否都表示存在连接
                if exists_matrix[i, k] > 0.5 and exists_matrix[k, i] > 0.5:
                    G.add_edge(i, k)

        # 2. 提取最大连通组件 (保持不变)
        if G.number_of_nodes() == 0:
            return G

        try:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc_nodes).copy()
        except ValueError:
            return nx.Graph()

        return G_main

    # -----------------------------------------------------------------
    # --- 3. 模块化指标函数 (Indicators) ---
    # --- (您可以在这里自行添加新函数) ---
    # -----------------------------------------------------------------

    def _calculate_DoF(self, G):
        """(辅助函数) 计算平面机构的自由度。"""
        num_links = G.number_of_edges()
        num_joints = G.number_of_nodes()
        if num_links < 3 or num_joints < 3:
            return -99
        # M = 3 * (L-1) - 2*J
        dof = 3 * (num_links - 1) - 2 * num_joints
        return dof

    def _check_connectivity(self, G):
        """
        指标1: 检查基本连通性和复杂性.
        这是一个 "硬约束", 失败会返回 -1.0。
        """
        # (nx.is_connected 已经在 _build_graph 中隐式处理了)

        # 约束: 节点数必须大于 3 (不能是三角形)
        if G.number_of_nodes() < 4:
            return -1.0  # 失败 (硬惩罚)

        return 1.0  # 通过

    def _check_dof(self, G):
        """
        指标2: 检查自由度是否为 1.
        这是一个 "软约束"。
        """
        dof = self._calculate_DoF(G)

        if dof == 1:
            return 1.0  # 奖励 (DoF=1)
        else:
            return -0.5  # 惩罚 (卡死或欠约束)

    def _check_topology_similarity(self, G):
        """
        指标3: 检查拓扑相似性 (偏好).
        这是一个 "奖励加成", 失败只返回 0。
        """
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # 偏好: 4杆机构
        if num_nodes == 4 and num_edges == 4:
            return 1.0  # 奖励加成

        return 0.0  # 中立 (不是4杆, 但不惩罚)

    # --- (新!) 指标4: 节点数量惩罚 ---
    def _check_node_count_penalty(self, G):
        """
        指标4: 惩罚节点数量与目标值 (e.g., 4) 的偏差.
        这是一个 "软约束" 或 "偏好塑造" 信号.
        """
        num_nodes = G.number_of_nodes()
        target_nodes = 4  # 您可以将其也放入 config 中

        # 使用二次惩罚 (偏差越大, 惩罚指数增长)
        # 我们希望 num_nodes == target_nodes 时惩罚为 0
        # 惩罚范围可以是 [0, -infinity], 我们需要将其映射到一个合理范围, e.g., [0, -1]

        deviation = abs(num_nodes - target_nodes)

        # 简单的线性惩罚示例:
        # penalty = -0.1 * deviation # 每偏离1个节点, 扣 0.1 分

        # 更强的二次惩罚示例 (归一化到 [-1, 0]):
        # max_deviation = self.max_nodes - target_nodes # 最大可能偏差
        # penalty = - (deviation / max_deviation) ** 2

        # 我们选择一个简单的二次惩罚，不进行严格归一化
        penalty_scale = 0.05  # 调整这个值来控制惩罚强度
        penalty = - penalty_scale * (deviation ** 2)

        # 确保惩罚不会超过 -1.0 (虽然不太可能)
        penalty = max(-1.0, penalty)

        # 返回值: 0.0 (如果恰好是4个节点) 或 负数 (如果偏离)
        return penalty
