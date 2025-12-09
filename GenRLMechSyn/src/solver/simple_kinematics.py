# src/solver/simple_kinematics.py
import torch
import math
from .utils import get_dh_matrix
import networkx as nx


def _build_extended_path(structure, raw_path):
    """
    构建扩展路径 (Padding)，用于处理任务约束的边界条件。
    返回: [Ghost_Prev, Start_Node, ..., End_Node, Ghost_Next]
    其中 Ghost 节点可能为 None。
    """
    if not raw_path:
        return None

    start_node = raw_path[0]
    end_node = raw_path[-1]
    path_set = set(raw_path)

    # 1. 找起点的前驱 (最小非路径邻居)
    # structure[u, :, 0] > 0.5 是邻居掩码
    start_neighbors = torch.nonzero(structure[start_node, :, 0] > 0.5).view(-1).tolist()
    valid_prev = [n for n in start_neighbors if n not in path_set]
    ghost_prev = min(valid_prev) if valid_prev else None

    # 2. 找终点的后继 (最大非路径邻居)
    end_neighbors = torch.nonzero(structure[end_node, :, 0] > 0.5).view(-1).tolist()
    valid_next = [n for n in end_neighbors if n not in path_set]
    ghost_next = max(valid_next) if valid_next else None

    # 3. 拼接
    return [ghost_prev] + raw_path + [ghost_next]


# ==============================================================================
# 1. 基础运动学计算 (计算螺旋轴)
# ==============================================================================

def compute_all_joint_screws(structure, joint_angles, base_node=0):
    """
    计算所有关节的瞬时螺旋轴。

    Args:
        structure: (N, N, 5) 结构张量
        joint_angles: (N, N) 关节锚点矩阵 (Anchor Angles)
        base_node: 基座节点索引

    Returns:
        all_screws: (N, 6) 每个节点的单位螺旋轴
    """
    device = structure.device
    N = structure.shape[0]
    transforms_map = {}
    screws_map = {}
    visited = [False] * N

    # 初始化基座
    T_base = torch.eye(4, device=device)
    transforms_map[base_node] = T_base
    visited[base_node] = True

    # BFS 队列: (current_node, off_in_u, q_in_u)
    # q_in_u: 进入该节点时的相位锚点
    queue = [(base_node, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))]

    TWO_PI = 2 * math.pi

    head = 0
    while head < len(queue):
        u, off_in_u, q_in_u = queue[head]
        head += 1

        T_global_u = transforms_map[u]

        # --- 计算当前节点的螺旋 (Screw) ---
        R_u = T_global_u[:3, :3]
        p_u = T_global_u[:3, 3]
        z_local = torch.tensor([0.0, 0.0, 1.0], device=device)
        z_axis = R_u @ z_local

        j_type_val = torch.max(structure[u, :, 1])
        is_R = (j_type_val > 0.0).float()
        is_P = 1.0 - is_R

        w = is_R * z_axis
        v_part = torch.linalg.cross(p_u, z_axis)
        v = is_R * v_part + is_P * z_axis
        screw_u = torch.cat([w, v], dim=0)
        screws_map[u] = screw_u

        # --- 传播到邻居 ---
        neighbors = torch.nonzero(structure[u, :, 0] > 0.5).squeeze(1)
        for v_idx in neighbors:
            v = v_idx.item()
            if not visited[v]:
                params = structure[u, v]
                a = torch.abs(params[2])
                alpha = params[3] % TWO_PI

                # 1. 几何参数差分
                off_out = params[4]
                delta_offset = off_out - off_in_u

                # 2. 变量参数差分 (Anchor Difference)
                # q_out - q_in
                q_out = joint_angles[u, v]
                delta_q = q_out - q_in_u

                # 3. 分配 DH 参数
                theta_val = is_R * delta_q + is_P * delta_offset
                d_val = is_R * delta_offset + is_P * delta_q

                T_step = get_dh_matrix(a, alpha, d_val, theta_val)
                T_global_v = T_global_u @ T_step
                transforms_map[v] = T_global_v
                visited[v] = True

                # 准备下一跳
                off_in_v = structure[v, u, 4]
                q_in_v = joint_angles[v, u]

                queue.append((v, off_in_v, q_in_v))

    # 组装结果
    screw_list = []
    zero_screw = torch.zeros(6, device=device)
    for i in range(N):
        if i in screws_map:
            screw_list.append(screws_map[i])
        else:
            screw_list.append(zero_screw)

    all_screws = torch.stack(screw_list)
    return all_screws, None


# ==============================================================================
# 2. 锚点速度求解器 (Anchor Velocity Solver)
# ==============================================================================

def solve_anchor_system(structure, q_current, loops, extended_task_path=None, target_twist=None, target_mask=None,
                        return_spectrum=False):
    """
    构建并求解锚点速度系统。

    [修正] 统一变量名为 extended_task_path，解决 NameError。
    """
    device = structure.device
    num_nodes = structure.shape[0]
    num_vars_full = num_nodes * num_nodes

    # --- 0. 动态确定基座 ---
    active_nodes = set()
    if loops:
        for loop in loops: active_nodes.update(loop)

    # [修正] 使用 extended_task_path 提取活跃节点 (排除 None)
    if extended_task_path:
        active_nodes.update([n for n in extended_task_path if n is not None])

    base_node = min(active_nodes) if active_nodes else 0

    # 1. 获取螺旋
    all_screws, _ = compute_all_joint_screws(structure, q_current, base_node=base_node)

    # 2. 构建全尺寸 K 矩阵和 b 向量
    num_loops = len(loops)
    # [修正] 检查 extended_task_path
    has_task = (target_twist is not None and target_mask is not None and extended_task_path is not None)

    if has_task and target_twist.dim() > 1:
        raise ValueError("仅支持单任务")

    total_rows = 6 * (num_loops + (1 if has_task else 0))
    K_full = torch.zeros((total_rows, num_vars_full), device=device)
    b = torch.zeros(total_rows, device=device)

    current_row = 0

    # --- A. 填充闭环约束 ---
    for loop_nodes in loops:
        L = len(loop_nodes)
        for i in range(L):
            curr = loop_nodes[i]
            next_node = loop_nodes[(i + 1) % L]
            prev_node = loop_nodes[(i - 1 + L) % L]

            screw = all_screws[curr]
            col_out = curr * num_nodes + next_node
            col_in = curr * num_nodes + prev_node

            K_full[current_row: current_row + 6, col_out] += screw
            K_full[current_row: current_row + 6, col_in] -= screw
        current_row += 6

    # --- B. 填充任务约束 ---
    if has_task:
        K_path = torch.zeros((6, num_vars_full), device=device)

        # [修正] 遍历 extended_task_path (含 Ghost 节点)
        # 结构: [Ghost_Prev, Start, ..., End, Ghost_Next]
        # 遍历中间的实体节点，索引从 1 到 len-2
        for i in range(1, len(extended_task_path) - 1):
            curr = extended_task_path[i]
            prev_node = extended_task_path[i - 1]  # 可能为 None
            next_node = extended_task_path[i + 1]  # 可能为 None

            screw = all_screws[curr]

            # 出边 (去往 Next)
            if next_node is not None:
                col_out = curr * num_nodes + next_node
                K_path[:, col_out] += screw

            # 入边 (来自 Prev)
            if prev_node is not None:
                col_in = curr * num_nodes + prev_node
                K_path[:, col_in] -= screw

        row_mask = (target_mask > 0.5).float().view(6, 1)
        K_full[current_row: current_row + 6, :] = K_path * row_mask
        b[current_row: current_row + 6] = target_twist * (target_mask > 0.5).float()

    # =================================================================
    # [核心] 矩阵瘦身 (Slimming Down)
    # =================================================================

    valid_mask = (structure[:, :, 0] > 0.5).view(-1)
    if not valid_mask.any():
        dummy_loss = torch.sum(1.0 - structure[:, :, 0]) * 100.0
        return K_full, torch.zeros(num_vars_full, device=device), dummy_loss, None

    K_reduced = K_full[:, valid_mask]
    x_sol_full = torch.zeros(num_vars_full, device=device)
    spectrum = None

    # --- C. 统一求解 (SVD分解) ---
    try:
        if has_task:
            # 增广矩阵法 [K_red | -b]
            b_reduced = b.unsqueeze(1)
            K_aug = torch.cat([K_reduced, -b_reduced], dim=1)

            # SVD 分解
            # 必须使用 full_matrices=True 以便在欠定系统(Rows < Cols)中获取完整的零空间基向量
            U, S, Vh = torch.linalg.svd(K_aug, full_matrices=True)

            # 提取解向量 (最小奇异值对应的右奇异向量 -> Vh 的最后一行)
            v_min = Vh[-1, :]
            x_reduced = v_min[:-1]
            lambda_val = v_min[-1]

            # 归一化 (强制 lambda=1)
            if torch.abs(lambda_val) > 1e-6:
                x_reduced = x_reduced / lambda_val
            else:
                x_reduced = x_reduced * 0.0

            # 构造兼容的 Spectrum (升序特征值)
            # 1. 平方 (Sigma^2 = Eigenvalue)
            # 2. 翻转 (SVD是降序, EIGH是升序)
            # 3. 补零 (如果 Rows < Cols，SVD 只返回 Rows 个值，剩下的都是 0)
            num_vars_aug = K_aug.shape[1]
            s_squared = (S ** 2)

            # 补齐缺少的 0 特征值
            padding_len = num_vars_aug - len(s_squared)
            if padding_len > 0:
                spectrum = torch.cat([torch.zeros(padding_len, device=device), s_squared.flip(0)])
            else:
                spectrum = s_squared.flip(0)

            # 残差 = 最小奇异值的平方 (对应之前的 eigenvalue)
            residual = spectrum[0]

        else:
            # 零空间求解 (K_reduced)
            U, S, Vh = torch.linalg.svd(K_reduced, full_matrices=True)

            # 最小奇异值向量
            x_reduced = Vh[-1, :]

            # 构造兼容 Spectrum
            num_vars_red = K_reduced.shape[1]
            s_squared = (S ** 2)

            padding_len = num_vars_red - len(s_squared)
            if padding_len > 0:
                spectrum = torch.cat([torch.zeros(padding_len, device=device), s_squared.flip(0)])
            else:
                spectrum = s_squared.flip(0)

            residual = spectrum[0]

        # 映射回全尺寸
        x_sol_full.masked_scatter_(valid_mask, x_reduced)

    except Exception as e:
        # 梯度保护
        bad_gradient = torch.mean(torch.abs(K_full)) * 1000.0
        return K_full, torch.zeros(num_vars_full, device=device), bad_gradient, None

    # =================================================================
    # [核心] 消除节点级规范自由度
    # =================================================================
    x_matrix = x_sol_full.view(num_nodes, num_nodes)
    valid_mask_matrix = (structure[:, :, 0] > 0.5).float()

    row_sums = torch.sum(x_matrix * valid_mask_matrix, dim=1, keepdim=True)
    row_counts = torch.sum(valid_mask_matrix, dim=1, keepdim=True)
    row_means = row_sums / (row_counts + 1e-8)

    x_centered = x_matrix - row_means
    x_matrix_clean = x_centered * valid_mask_matrix
    x_sol_final = x_matrix_clean.view(-1)

    return K_full, x_sol_final, residual, spectrum


# ==============================================================================
# 3. 核心 Loss 计算 (基于锚点速度的一致性)
# ==============================================================================

def compute_motion_consistency_loss(structure, q_current, loops, path_to_ee,
                                    target_twists=None, target_masks=None, dt=1e-3):
    """
    基于 Anchor Velocity 的二阶全周一致性 Loss。

    逻辑:
    1. 针对每个任务模式，求解 T=0 时刻的一阶锚点速度 x0。
    2. 更新锚点位置 Q_new = Q + x0 * dt。
    3. 构建 T=dt 时刻的矩阵 K1。
    4. 计算漂移 Drift = (K1 * x0 - K0 * x0) / dt。
    5. 求解二阶加速度 x_ddot，并计算其与 x0 的法向偏差。
    """
    device = structure.device
    num_nodes = structure.shape[0]
    total_loss = torch.tensor(0.0, device=device)

    # 预构建扩展路径 (避免在循环中重复构建)
    extended_path = _build_extended_path(structure, path_to_ee)

    has_tasks = (target_twists is not None and target_masks is not None and len(target_twists) > 0)
    num_modes = target_twists.shape[0] if has_tasks else 1

    for k in range(num_modes):
        tgt_twist = target_twists[k] if has_tasks else None
        tgt_mask = target_masks[k] if has_tasks else None

        # 1. 求解 T=0 时刻的一阶锚点速度 x0
        # 返回: K, x, residual, spectrum
        K0, x0, residual0, _ = solve_anchor_system(
            structure, q_current, loops, extended_path, tgt_twist, tgt_mask
        )

        # 归一化 x0 (防止数值过大导致微分失效，或数值过小导致精度丢失)
        x_norm = torch.norm(x0)

        # 如果速度极小(死锁) 或者 任务残差过大(不可达)，直接惩罚并跳过二阶计算
        if x_norm < 1e-6 or (has_tasks and residual0 > 0.1):
            total_loss += 1.0
            continue

        x0 = x0 / x_norm

        # 2. 更新状态 (Q矩阵更新)
        # Anchor Velocity 是锚点位置的时间导数，直接叠加
        q_next = q_current + x0.view(num_nodes, num_nodes) * dt

        # 3. 构建 T=dt 时刻的矩阵 K1
        # 我们只需要 K1，不需要求解
        K1, _, _, _ = solve_anchor_system(
            structure, q_next, loops, extended_path, tgt_twist, tgt_mask
        )

        # 4. 计算漂移 (Drift)
        # Drift = d(Kx)/dt = (K_new * x - K_old * x) / dt
        # 对于有任务的情况 (Kx=b)，d(Kx-b)/dt = K_dot*x = 0，所以 drift 依然是衡量 K 变化的指标
        term1 = K1 @ x0
        term2 = K0 @ x0
        drift = (term1 - term2) / dt

        # 5. 求解二阶锚点加速度 x_ddot
        # 方程: K0 * x_ddot = -drift
        # 使用阻尼最小二乘求解线性方程组
        H = K0.T @ K0
        damping = 1e-4 * torch.eye(num_nodes * num_nodes, device=device)
        rhs = K0.T @ (-drift)

        try:
            x_ddot = torch.linalg.solve(H + damping, rhs)
        except:
            x_ddot = torch.zeros_like(x0)

        # 6. 一致性判据 (Consistency Metric)
        # 投影: 计算 x_ddot 在 x0 方向上的垂直分量
        proj = torch.dot(x_ddot, x0) * x0
        x_perp = x_ddot - proj

        # Loss: 漂移分量的模长
        mode_loss = torch.norm(x_perp)
        total_loss += mode_loss

    if has_tasks:
        return total_loss / num_modes
    else:
        return total_loss


# ==============================================================================
# 4. 其他 Loss 函数 (保持逻辑，适配接口)
# ==============================================================================

def compute_loop_errors(structure, joint_angles, loops):
    """
    [修改版] 支持 q_opt 为 (N, N) 矩阵。
    """
    device = structure.device
    total_error = torch.tensor(0.0, device=device)
    TWO_PI = 2 * math.pi

    for path in loops:
        T_cum = torch.eye(4, device=device)
        L = len(path)
        max_link_length = torch.tensor(0.0, device=device)

        for i in range(L):
            u = path[i]
            v = path[(i + 1) % L]
            prev = path[(i - 1 + L) % L]

            # 1. 提取结构参数
            params = structure[u, v]
            j_type = params[1]
            a = torch.abs(params[2])
            alpha = params[3] % TWO_PI
            max_link_length = torch.max(max_link_length, a)

            # 2. Offset 差分
            off_out = structure[u, v, 4]
            off_in = structure[u, prev, 4]
            delta_offset = off_out - off_in

            # 3. Anchor Difference
            q_out = joint_angles[u, v]
            q_in = joint_angles[u, prev]
            delta_q = q_out - q_in

            # 4. 分配变量
            is_R = (j_type > 0).float()
            is_P = 1.0 - is_R

            theta = is_R * delta_q + is_P * delta_offset
            d = is_R * delta_offset + is_P * delta_q

            # 5. 计算矩阵
            T_step = get_dh_matrix(a, alpha, d, theta)
            T_cum = T_cum @ T_step

        # 6. 计算误差
        pos_err_abs = torch.sum(T_cum[:3, 3] ** 2)
        scale_factor = max_link_length ** 2 + 1e-6
        pos_err_rel = pos_err_abs / scale_factor
        rot_err = torch.sum((T_cum[:3, :3] - torch.eye(3, device=device)) ** 2)

        total_error = total_error + pos_err_rel + rot_err

    return total_error


def compute_bennett_geometry_error(structure, loops):
    """
    [修改说明]
    1. 增加关节类型约束 (R副)。
    2. [新增] 增加偏移量约束: 要求 DH 参数 d = 0。
    3. 保持相对误差计算逻辑。
    """
    device = structure.device
    total_error = torch.tensor(0.0, device=device)
    TWO_PI = 2 * math.pi

    for path in loops:
        if len(path) != 4:
            continue

        # 1. 收集参数 & 检查关节类型 & 检查偏移量
        a_list = []
        alpha_list = []

        type_loss_accum = torch.tensor(0.0, device=device)
        offset_loss_accum = torch.tensor(0.0, device=device)

        # 用于归一化的最大杆长 (避免除以0)
        max_a = torch.tensor(1.0, device=device)

        L = 4
        for i in range(L):
            u = path[i]
            v = path[(i + 1) % L]  # 下一个节点 (出边)
            prev = path[(i - 1 + L) % L]  # 上一个节点 (入边)

            params = structure[u, v]

            # --- A. 关节类型检查 (R副) ---
            j_type = params[1]
            type_loss_accum = type_loss_accum + torch.relu(0.5 - j_type)

            # --- B. 提取几何参数 ---
            val_a = torch.abs(params[2])
            val_alpha = params[3] % TWO_PI

            # 更新最大杆长用于归一化
            max_a = torch.max(max_a, val_a)

            a_list.append(val_a)
            alpha_list.append(val_alpha)

            # --- C. [新增] 偏移量检查 (d=0) ---
            # Bennett 机构要求 DH 参数 d = 0
            # d = offset_out - offset_in
            off_out = params[4]
            off_in = structure[u, prev, 4]

            d_val = off_out - off_in

            # 累加 d^2
            offset_loss_accum = offset_loss_accum + (d_val ** 2)

        # 将 offset 误差归一化 (相对误差)
        # Loss = sum(d^2) / max_a^2
        offset_loss_rel = offset_loss_accum / (max_a ** 2 + 1e-6)

        a_vec = torch.stack(a_list)
        alpha_vec = torch.stack(alpha_list)

        # 2. 相对对称误差 (Relative Symmetry Error)

        # 杆长对称性
        diff_a_13 = (a_vec[0] - a_vec[2]) ** 2
        norm_a_13 = (a_vec[0].detach() + a_vec[2].detach()) ** 2 + 1e-6
        sym_loss_a = diff_a_13 / norm_a_13

        diff_a_24 = (a_vec[1] - a_vec[3]) ** 2
        norm_a_24 = (a_vec[1].detach() + a_vec[3].detach()) ** 2 + 1e-6
        sym_loss_a += diff_a_24 / norm_a_24

        # 角度对称性
        sym_loss_alpha = (alpha_vec[0] - alpha_vec[2]) ** 2 + (alpha_vec[1] - alpha_vec[3]) ** 2

        # 3. 相对 Bennett 比例误差
        sin_alpha = torch.sin(alpha_vec)

        # 组1
        term1_1 = a_vec[0] * sin_alpha[1]
        term1_2 = a_vec[1] * sin_alpha[0]
        ratio_err1_abs = (term1_1 - term1_2) ** 2
        ratio_err1_rel = ratio_err1_abs / (term1_1.detach() ** 2 + term1_2.detach() ** 2 + 1e-6)

        # 组2
        term2_1 = a_vec[1] * sin_alpha[2]
        term2_2 = a_vec[2] * sin_alpha[1]
        ratio_err2_abs = (term2_1 - term2_2) ** 2
        ratio_err2_rel = ratio_err2_abs / (term2_1.detach() ** 2 + term2_2.detach() ** 2 + 1e-6)

        # 4. 总误差汇总
        # [修改] 加入 offset_loss_rel
        # 权重分配:
        # - 关节类型: 10.0 (必须满足)
        # - 偏移量: 5.0 (必须为0，很重要)
        # - 几何对称/比例: 1.0 (优化目标)

        loop_error = (sym_loss_a + sym_loss_alpha + ratio_err1_rel + ratio_err2_rel) + \
                     (type_loss_accum * 10.0) + \
                     (offset_loss_rel * 5.0)

        total_error = total_error + loop_error

    return total_error


def compute_mobility_loss_eigen(structure, q, loops, num_dof=1, gap_threshold=0.01, require_exact_dof=False):
    """
    [修改版] 复用 solve_anchor_system 的计算结果。
    """
    device = structure.device
    num_nodes = structure.shape[0]

    # 1. 调用求解器，获取特征值谱 (Spectrum)
    # 注意：这里我们不需要任务 (target_twist=None)，只关心机构本身的拓扑属性
    _, _, _, spectrum = solve_anchor_system(
        structure, q, loops,
        extended_task_path=None, target_twist=None, target_mask=None,
        return_spectrum=True
    )

    # 如果求解失败返回 None
    if spectrum is None:
        return torch.tensor(100.0, device=device)

    # 2. 这里的 spectrum 已经是 K_reduced^T @ K_reduced 的特征值了
    # 直接使用即可
    eigenvalues = spectrum

    # 3. 定义目标零空间维度
    # Total Nullity = N (Gauge) + F (Physical)
    target_zero_count = num_nodes + num_dof

    if len(eigenvalues) <= target_zero_count:
        return torch.tensor(10.0, device=device)

    # --- A. 零空间 Loss ---
    target_zeros = eigenvalues[:target_zero_count]
    loss_zeros = torch.sum(torch.abs(target_zeros))

    # --- B. Gap Loss ---
    loss_gap = torch.tensor(0.0, device=device)
    if require_exact_dof:
        next_eig = eigenvalues[target_zero_count]
        loss_gap = torch.relu(gap_threshold - next_eig)

    total_loss = loss_zeros * 100.0 + loss_gap * 10.0
    return total_loss


def compute_task_loss_eigen(structure, q, loops, G_graph, config_ee_node, target_twists, target_masks):
    """
    [修改版] 任务残差 Loss。

    修正逻辑：
    1. 不再使用 solve_anchor_system 返回的 residual (spectrum[0])，因为它被规范自由度占据永远为0。
    2. 而是提取 spectrum，剔除前 N 个规范自由度。
    3. 最小化 spectrum[num_nodes] (即第 N+1 个特征值)。
    """
    device = structure.device
    num_nodes = structure.shape[0]
    loss_total = torch.tensor(0.0, device=device)

    # 1. 确定路径
    graph_nodes = list(G_graph.nodes())
    if not graph_nodes: return loss_total

    base_node = min(graph_nodes)
    ee_node = config_ee_node if config_ee_node in graph_nodes else max(graph_nodes)

    try:
        raw_path = nx.shortest_path(G_graph, source=base_node, target=ee_node)
    except:
        return torch.tensor(100.0, device=device)

    # 2. 构建扩展路径
    extended_path = _build_extended_path(structure, raw_path)

    num_patterns = target_twists.shape[0]

    # 3. 遍历每个任务模式
    for k in range(num_patterns):
        tgt = target_twists[k]
        mask = target_masks[k]

        if mask.bool().any():
            # 调用求解器，请求返回谱 (Spectrum)
            _, _, _, spectrum = solve_anchor_system(
                structure, q, loops,
                extended_task_path=extended_path,
                target_twist=tgt,
                target_mask=mask,
                return_spectrum=True
            )

            # 异常保护
            if spectrum is None:
                loss_total += 10.0
                continue

            # [核心修正]
            # 锚点系统必然存在 N 个规范自由度 (特征值为0)
            # 我们要检查的是：是否存在第 N+1 个零空间向量 (代表任务解)
            # 所以我们要优化的是 spectrum[num_nodes]

            # 检查谱的长度是否足够
            target_idx = num_nodes
            if len(spectrum) > target_idx:
                # 取第 N+1 个特征值 (注意 spectrum 是升序排列的)
                # 这个值越小，说明 [K|-b] 越接近存在一个非平凡的零空间解
                val = spectrum[target_idx]

                # 加上绝对值防止极微小的负数噪声
                loss_total += torch.abs(val)
            else:
                # 这种情况极少发生 (除非有效边数极少)，给一个惩罚
                loss_total += 1.0

    return loss_total * 50.0  # 权重系数
