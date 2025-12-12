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

def solve_anchor_system(structure, q_current, loops, extended_task_path=None, target_twist=None,
                        target_mask=None, return_spectrum=False):
    """
    构建并求解锚点速度系统。
    返回:
        edge_to_col: 字典 {(u,v): col_idx, ...}
        K_compact:   紧凑 Jacobian
        x_compact:   紧凑解向量 (None if SVD fails)
        residual:    任务残差
        spectrum:    奇异值谱
    """
    device = structure.device

    # 1. 确定计算所需的节点和 Screw
    active_nodes = set()
    for loop in loops: active_nodes.update(loop)
    if extended_task_path:
        active_nodes.update([n for n in extended_task_path if n is not None and n >= 0])

    if not active_nodes:
        return None, None, None, None, None

    # 确定基座
    if extended_task_path and len(extended_task_path) > 1 and extended_task_path[1] is not None:
        base_node = extended_task_path[1]
    else:
        base_node = min(active_nodes)

    all_screws, _ = compute_all_joint_screws(structure, q_current, base_node)

    # 2. 建立变量映射 (Edge to Column)
    involved_edges_set = set()
    # A. Loops
    for loop in loops:
        L = len(loop)
        for i in range(L):
            u, v = loop[i], loop[(i + 1) % L]
            involved_edges_set.add(tuple(sorted((u, v))))
    # B. Task Path
    if extended_task_path:
        for i in range(len(extended_task_path) - 1):
            u, v = extended_task_path[i], extended_task_path[i + 1]
            if u is not None and v is not None and u >= 0 and v >= 0:
                involved_edges_set.add(tuple(sorted((u, v))))

    # C. 构建映射
    edge_to_col = {}
    current_col = 0
    for u, v in sorted(list(involved_edges_set)):
        edge_to_col[(u, v)] = current_col;
        current_col += 1
        edge_to_col[(v, u)] = current_col;
        current_col += 1

    num_vars_reduced = current_col
    if num_vars_reduced == 0:
        return None, None, None, None, None

    # 3. 构建 Jacobian
    num_loops = len(loops)
    has_task = (target_twist is not None)
    total_rows = 6 * (num_loops + (1 if has_task else 0))

    K_compact = torch.zeros((total_rows, num_vars_reduced), device=device)
    b = torch.zeros(total_rows, device=device)
    current_row = 0

    # --- 填充 Loop 约束 ---
    for loop in loops:
        L = len(loop)
        for i in range(L):
            curr, next_n, prev = loop[i], loop[(i + 1) % L], loop[(i - 1 + L) % L]
            screw = all_screws[curr]

            if (curr, next_n) in edge_to_col:
                K_compact[current_row:current_row + 6, edge_to_col[(curr, next_n)]] += screw
            if (curr, prev) in edge_to_col:
                K_compact[current_row:current_row + 6, edge_to_col[(curr, prev)]] -= screw
        current_row += 6

    # --- 填充 Task 约束 ---
    if has_task and extended_task_path:
        row_slice = slice(current_row, current_row + 6)
        for i in range(1, len(extended_task_path) - 1):
            curr = extended_task_path[i]
            prev_n, next_n = extended_task_path[i - 1], extended_task_path[i + 1]
            screw = all_screws[curr]

            if next_n is not None and next_n >= 0 and (curr, next_n) in edge_to_col:
                K_compact[row_slice, edge_to_col[(curr, next_n)]] += screw
            if prev_n is not None and prev_n >= 0 and (curr, prev_n) in edge_to_col:
                K_compact[row_slice, edge_to_col[(curr, prev_n)]] -= screw

        if target_mask is not None:
            K_compact[row_slice, :] *= target_mask.view(6, 1)
            b[row_slice] = target_twist * target_mask
        else:
            b[row_slice] = target_twist

    # 4. SVD 求解
    if has_task:
        K_aug = torch.cat([K_compact, -b.unsqueeze(1)], dim=1)
    else:
        K_aug = K_compact

    try:
        U, S, Vh = torch.linalg.svd(K_aug, full_matrices=False)

        # Spectrum
        num_vars_aug = K_aug.shape[1]
        full_S = torch.zeros(num_vars_aug, dtype=S.dtype, device=device)
        full_S[:S.shape[0]] = S
        spectrum = torch.flip(full_S, dims=[0])

        # 提取解向量 (最小奇异值对应的 Vh 行)
        v_min = Vh[-1, :]

        if has_task:
            lambda_val = v_min[-1]
            x_vars = v_min[:-1]
            # 归一化 lambda=1
            if torch.abs(lambda_val) > 1e-6:
                x_vars = x_vars / lambda_val
            else:
                x_vars = torch.zeros_like(x_vars)
            residual = spectrum[0]
        else:
            x_vars = v_min
            residual = spectrum[0]

        return edge_to_col, K_compact, x_vars, residual, spectrum

    except Exception as e:
        return None, None, None, torch.tensor(100.0, device=device), None


# ==============================================================================
# 3. 核心 Loss 计算 (基于锚点速度的一致性)
# ==============================================================================

def compute_instantaneous_check_loss(structure, q_current, loops, path_to_ee,
                                     target_twists=None, target_masks=None, dt=1e-3):
    """
    [IDOF 检测版] 运动可持续性检查

    原理：
    构造"虚拟环路"（即包含任务约束的雅可比矩阵），检查二阶漂移（Drift）是否落在
    雅可比矩阵（K）的列空间内。

    如果是瞬时运动 (IDOF)，Drift 将无法被 K 补偿，产生巨大的投影残差。
    """
    device = structure.device
    total_loss = torch.tensor(0.0, device=device)

    # 1. 扩展路径 (用于构建包含任务的 K 矩阵)
    extended_path = _build_extended_path(structure, path_to_ee)

    has_tasks = (target_twists is not None and target_masks is not None and len(target_twists) > 0)
    num_modes = target_twists.shape[0] if has_tasks else 1

    for k in range(num_modes):
        tgt_twist = target_twists[k] if has_tasks else None
        tgt_mask = target_masks[k] if has_tasks else None

        # --- 步骤 1: 获取当前构型的 K 和 x0 ---
        # 这里的 K_curr 实际上就是包含了"虚拟环路约束"（任务约束）的雅可比矩阵
        mapping, K_curr, x0, resid0, _ = solve_anchor_system(
            structure, q_current, loops, extended_path, tgt_twist, tgt_mask
        )

        # 基础检查：如果是死锁或当前位置就不闭合，直接重罚
        if x0 is None:
            continue

        # 归一化 x0 (单位速度，消除速度大小对 Drift 幅度的影响)
        x_norm = torch.norm(x0)
        if x_norm < 1e-6:
            continue
        x0 = x0 / x_norm

        # --- 步骤 2: 计算二阶漂移 (The "Bill") ---
        # 我们使用有限差分来通过 PyTorch 自动计算李括号项 (J_dot * q_dot)
        # 这比手动实现 _lie_bracket 更通用，且能自动处理复杂的螺旋轴变化

        # 2.1 模拟向前走极小的一步
        q_next = q_current.clone()
        for (u, v), col_idx in mapping.items():
            if col_idx < len(x0):
                q_next[u, v] += x0[col_idx] * dt

        # 2.2 获取新位置的 K (无需解方程，只要矩阵)
        _, K_next, _, _, _ = solve_anchor_system(
            structure, q_next, loops, extended_path, tgt_twist, tgt_mask
        )

        if K_next is None:
            total_loss += 10.0;
            continue

        # 2.3 计算漂移向量 Drift = (K_next - K_curr) * x0 / dt
        # 物理含义：保持关节速度不变时，约束方程产生的破坏速度
        drift_vec = (K_next @ x0 - K_curr @ x0) / dt

        # --- 步骤 3: 投影相容性测试 (The "Payment") ---
        # 检查方程 K_curr * alpha = -drift 是否有解
        # 如果有解，说明可以通过调整关节加速度 alpha 来消除漂移 -> 运动是可持续的
        # 如果无解（残差大），说明是瞬时运动 -> IDOF

        # 使用伪逆进行投影: Projection = K * K_pinv
        # Residual = (I - Projection) * drift
        #          = drift - K * (K_pinv * drift)

        # rcond=1e-3 用于忽略极小的奇异值噪声
        K_pinv = torch.linalg.pinv(K_curr, rcond=1e-3)

        # 尝试求解加速度 (best effort solution)
        alpha_sol = K_pinv @ (-drift_vec)

        # 实际能补偿的漂移
        compensated_drift = K_curr @ alpha_sol

        # --- 步骤 4: 计算残差 Loss ---
        # residual_vec 代表了"无法被机构几何结构消解的二阶漂移"
        # 对于平行四边形：几何结构完美，drift 虽大但完全在 K 的列空间内，Residual ≈ 0
        # 对于瞬时机构：drift 指向 K 列空间之外，Residual >> 0

        residual_vec = (-drift_vec) - compensated_drift
        loss_idof = torch.norm(residual_vec)

        # 标准化：除以 drift 的模长 (Ratio)
        drift_norm = torch.norm(drift_vec)
        if drift_norm > 1e-6:
            loss_ratio = loss_idof / drift_norm
        else:
            loss_ratio = 0.0  # 几乎没有漂移，说明是直线机构，完美

        total_loss += loss_ratio

    return total_loss / num_modes if has_tasks else total_loss


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
    _, _, _, _, spectrum = solve_anchor_system(
        structure, q, loops, None, None, None, return_spectrum=True
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
            _, _, _, _, spectrum = solve_anchor_system(
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
