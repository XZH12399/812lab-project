# src/solver/simple_kinematics.py
import torch
import math
from .utils import get_dh_matrix
import networkx as nx


def compute_loop_errors(structure, joint_angles, loops):
    """
    计算闭环误差 (纯 PyTorch 实现，支持自动微分)
    [修改说明]
    1. 使用 abs/mod 保证物理参数合法。
    2. [修改] 使用相对误差：位置误差除以 (最大杆长)^2。
       这比除以总杆长更严格，能避免误差被长周长稀释。
    """
    device = structure.device
    total_error = torch.tensor(0.0, device=device)
    TWO_PI = 2 * math.pi

    for path in loops:
        T_cum = torch.eye(4, device=device)
        L = len(path)

        # [新增] 追踪环路中的最大杆长
        max_link_length = torch.tensor(0.0, device=device)

        for i in range(L):
            u = path[i]
            v = path[(i + 1) % L]
            prev = path[(i - 1 + L) % L]

            # 1. 提取参数
            params = structure[u, v]
            j_type = params[1]

            # 物理约束处理
            a = torch.abs(params[2])
            alpha = params[3] % TWO_PI

            # [核心修改] 更新最大杆长
            # 使用 torch.max 保持梯度流 (Subgradient)
            max_link_length = torch.max(max_link_length, a)

            # 2. Offset 差分
            off_out = structure[u, v, 4]
            off_in = structure[u, prev, 4]
            delta = off_out - off_in

            # 3. 提取变量
            q = joint_angles[u]

            # 4. 分配变量
            is_R = (j_type > 0).float()
            is_P = 1.0 - is_R

            theta = is_R * q + is_P * delta
            d = is_R * delta + is_P * q

            # 5. 计算矩阵
            T_step = get_dh_matrix(a, alpha, d, theta)
            T_cum = T_cum @ T_step

        # 6. 计算误差
        # 位置误差 (绝对值平方)
        pos_err_abs = torch.sum(T_cum[:3, 3] ** 2)

        # [核心修改] 转换为相对位置误差
        # 除以 (最大杆长)^2
        # 如果最大杆长接近0 (虽然我们有约束防止它发生)，加 epsilon 防止除以0
        scale_factor = max_link_length ** 2 + 1e-6
        pos_err_rel = pos_err_abs / scale_factor

        # 姿态误差 (旋转矩阵本身就是归一化的)
        rot_err = torch.sum((T_cum[:3, :3] - torch.eye(3, device=device)) ** 2)

        # 总误差
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


def compute_all_joint_screws(structure, joint_angles, base_node=0):
    """
    [修正版] 避免 In-place 操作，修复 RuntimeError。
    使用字典存储中间变换矩阵，最后再 stack 成张量。
    """
    device = structure.device
    N = structure.shape[0]

    # 1. 初始化容器 (使用字典代替 Tensor 以避免 In-place 修改)
    # transforms_map: {node_index: T_global_tensor}
    transforms_map = {}

    # 存储结果螺旋的字典
    screws_map = {}

    # 标记是否已计算
    visited = [False] * N

    # 2. 设置基座 (Base)
    T_base = torch.eye(4, device=device)
    transforms_map[base_node] = T_base
    visited[base_node] = True

    # 3. BFS 队列
    queue = [(base_node, torch.tensor(0.0, device=device))]
    TWO_PI = 2 * math.pi

    head = 0
    while head < len(queue):
        u, off_in_u = queue[head]
        head += 1

        # 从字典中获取 u 的全局位姿 (这是独立的 tensor，不是 slice)
        T_global_u = transforms_map[u]

        # --- A. 计算当前节点 u 的螺旋 ---
        R_u = T_global_u[:3, :3]
        p_u = T_global_u[:3, 3]

        z_local = torch.tensor([0.0, 0.0, 1.0], device=device)
        z_axis = R_u @ z_local

        # 获取关节类型 (取行最大值作为标识)
        j_type_val = torch.max(structure[u, :, 1])
        is_R = (j_type_val > 0.0).float()
        is_P = 1.0 - is_R

        # 构造螺旋
        w = is_R * z_axis
        v_part = torch.linalg.cross(p_u, z_axis)
        v = is_R * v_part + is_P * z_axis

        screw_u = torch.cat([w, v], dim=0)
        screws_map[u] = screw_u

        # --- B. 传播到邻居 ---
        neighbors = torch.nonzero(structure[u, :, 0] > 0.5).squeeze(1)

        for v_idx in neighbors:
            v = v_idx.item()
            if not visited[v]:
                # 计算 T_{u->v}
                params = structure[u, v]
                a = torch.abs(params[2])
                alpha = params[3] % TWO_PI

                off_out = params[4]
                d = off_out - off_in_u

                q = joint_angles[u]
                theta_val = is_R * q
                d_val = d + is_P * q

                T_step = get_dh_matrix(a, alpha, d_val, theta_val)

                # 计算 v 的全局位姿 (创建新 Tensor)
                T_global_v = T_global_u @ T_step

                # 存入字典 (安全操作)
                transforms_map[v] = T_global_v
                visited[v] = True

                off_in_v = structure[v, u, 4]
                queue.append((v, off_in_v))

    # --- C. 重组为张量 ---
    # 将字典转回 (N, 6) 的张量
    screw_list = []
    # 也可以顺便返回 transforms 张量用于 debug，这里只需 screws

    zero_screw = torch.zeros(6, device=device)

    for i in range(N):
        if i in screws_map:
            screw_list.append(screws_map[i])
        else:
            # 对于未连接的节点，填充 0
            screw_list.append(zero_screw)

    # 使用 stack 保持梯度流
    all_screws = torch.stack(screw_list)

    # global_transforms 如果不需要可以返回 None，或者同样用 stack 组装
    return all_screws, None


def compute_mobility_loss_eigen(structure, q, loops, num_dof=1, gap_threshold=0.005):
    """
    计算全局可动性 Loss。
    Args:
        gap_threshold: 谱间隙阈值。第 K+1 个特征值必须大于此值。
    """
    device = structure.device

    # ... (1. 动态确定基座 ... 2. 构建雅可比 ... 3. 特征值分解 保持不变) ...

    # 1. 动态确定基座
    active_nodes = set()
    for loop in loops:
        active_nodes.update(loop)
    if not active_nodes: return torch.tensor(0.0, device=device)
    base_node = min(active_nodes)

    all_screws, _ = compute_all_joint_screws(structure, q, base_node=base_node)

    constraint_rows = []
    num_nodes = structure.shape[0]
    for loop_nodes in loops:
        J_loop = torch.zeros((6, num_nodes), device=device)
        for node_idx in loop_nodes:
            J_loop[:, node_idx] = all_screws[node_idx]
        constraint_rows.append(J_loop)
    J_global = torch.cat(constraint_rows, dim=0)

    G_mat = J_global.T @ J_global
    eigenvalues = torch.linalg.eigvalsh(G_mat)  # 升序

    # 4. 计算 Loss

    # A. 必须为 0 的部分 (Target DOFs)
    target_zero_eigs = eigenvalues[:num_dof]
    loss_zeros = torch.sum(torch.abs(target_zero_eigs))

    # B. 必须不为 0 的部分 (Spectral Gap)
    loss_gap = torch.tensor(0.0, device=device)
    if len(eigenvalues) > num_dof:
        next_eig = eigenvalues[num_dof]

        # [修改] 使用传入的 gap_threshold
        # 我们希望 next_eig >= gap_threshold
        loss_gap = torch.relu(gap_threshold - next_eig)

    # 5. 总 Loss
    total_loss = loss_zeros * 100.0 + loss_gap * 10.0

    return total_loss


def compute_task_loss_eigen(structure, q, loops, G_graph, config_ee_node, target_twists, target_masks):
    """
    [增强版 v2.0] 计算任务 Loss (全系统增广矩阵法)。
    同时考虑闭环约束和任务目标，确保满足任务的关节速度同时也满足闭环条件。

    构建矩阵:
    [ J_loops       |  0   ]
    [ J_path_masked | Target ]
    """
    device = structure.device
    loss_total = torch.tensor(0.0, device=device)

    # 1. 动态确定基座和末端
    graph_nodes = list(G_graph.nodes())
    if not graph_nodes: return loss_total

    base_node = min(graph_nodes)
    if config_ee_node in graph_nodes:
        final_ee_node = config_ee_node
    else:
        final_ee_node = max(graph_nodes)

    # 2. 计算全局螺旋
    all_screws, _ = compute_all_joint_screws(structure, q, base_node=base_node)

    # 3. 构建闭环雅可比 J_loops (6L, N)
    # 这是为了保证求出的速度解满足机构的物理约束
    num_nodes = structure.shape[0]
    constraint_rows = []
    for loop_nodes in loops:
        J_loop = torch.zeros((6, num_nodes), device=device)
        for node_idx in loop_nodes:
            J_loop[:, node_idx] = all_screws[node_idx]
        constraint_rows.append(J_loop)

    if constraint_rows:
        J_loops = torch.cat(constraint_rows, dim=0)
    else:
        # 如果没有环(虽然会被过滤)，给一个空矩阵
        J_loops = torch.zeros((0, num_nodes), device=device)

    # 4. 构建路径雅可比 J_path (6, N)
    try:
        path_to_ee = nx.shortest_path(G_graph, source=base_node, target=final_ee_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return loss_total

    path_screws = [all_screws[u] for u in path_to_ee]
    J_path = torch.stack(path_screws, dim=1)  # (6, M)

    # 需要将 J_path 映射回 (6, N) 的全尺寸矩阵，以便与 J_loops 对齐
    J_path_full = torch.zeros((6, num_nodes), device=device)
    for i, u in enumerate(path_to_ee):
        J_path_full[:, u] = J_path[:, i]

    # 5. 遍历目标模式，构建增广矩阵并计算特征值
    num_patterns = target_twists.shape[0]

    for k in range(num_patterns):
        target_val = target_twists[k]
        mask = target_masks[k]
        mask_bool = mask.bool()

        if mask_bool.any():
            # --- A. 准备子矩阵 ---
            # 任务部分: 只取 Mask 选中的行
            J_path_masked = J_path_full[mask_bool, :]  # (D, N)
            target_masked = target_val[mask_bool].unsqueeze(1)  # (D, 1)

            # --- B. 构建全系统矩阵 ---
            # 我们要构建:
            # [ J_loops | 0 ]  <-- 闭环约束 (必须满足)
            # [ J_path  | t ]  <-- 任务约束

            # 上半部分: [J_loops, 0]
            if J_loops.size(0) > 0:
                zeros_col = torch.zeros((J_loops.size(0), 1), device=device)
                top_block = torch.cat([J_loops, zeros_col], dim=1)  # (6L, N+1)
            else:
                top_block = torch.empty((0, num_nodes + 1), device=device)

            # 下半部分: [J_path_masked, target_masked]
            bottom_block = torch.cat([J_path_masked, target_masked], dim=1)  # (D, N+1)

            # 拼接
            M_total = torch.cat([top_block, bottom_block], dim=0)

            # --- C. 计算特征值 ---
            G_aug = M_total.T @ M_total
            evals_aug = torch.linalg.eigvalsh(G_aug)

            # 最小特征值应为 0
            # 这意味着存在一组 (q_dot, c) 同时满足闭环和任务
            loss_total += torch.abs(evals_aug[0])

    return loss_total * 50.0  # 权重


def _get_jacobians_and_screws(structure, q, loops, path_to_ee, base_node=0):
    """
    辅助函数：计算闭环雅可比 J_loop、完整路径雅可比 J_path_full 以及所有螺旋。
    """
    device = structure.device
    num_nodes = structure.shape[0]

    # 1. 计算所有关节的瞬时螺旋
    all_screws, _ = compute_all_joint_screws(structure, q, base_node=base_node)

    # 2. 构建闭环约束雅可比 J_loop (6L x N)
    loop_rows = []
    if loops:
        for loop_nodes in loops:
            J_sub = torch.zeros((6, num_nodes), device=device)
            for node_idx in loop_nodes:
                J_sub[:, node_idx] = all_screws[node_idx]
            loop_rows.append(J_sub)
        J_loop = torch.cat(loop_rows, dim=0)
    else:
        J_loop = torch.zeros((0, num_nodes), device=device)

    # 3. 构建路径雅可比 J_path_full (6 x N)
    # 每一列对应路径上该节点的螺旋，不在路径上的为0
    J_path_full = torch.zeros((6, num_nodes), device=device)
    if path_to_ee:
        for i, node_idx in enumerate(path_to_ee):
            # 注意：all_screws[node_idx] 是 (6,)
            J_path_full[:, node_idx] = all_screws[node_idx]

    return J_loop, J_path_full, all_screws


def compute_motion_consistency_loss(structure, q_current, loops, path_to_ee,
                                    target_twists=None, target_masks=None, dt=1e-3):
    """
    计算二阶全周运动一致性 Loss (修复版)。

    [修复] 将 torch.linalg.eigvalsh 改为 torch.linalg.eigh 以正确获取特征向量。
    """
    device = structure.device
    base_node = min(path_to_ee) if path_to_ee else 0
    num_nodes = structure.shape[0]

    # --- 1. 准备 T=0 时刻的雅可比 ---
    J_loop_0, J_path_0, _ = _get_jacobians_and_screws(structure, q_current, loops, path_to_ee, base_node)

    total_loss = torch.tensor(0.0, device=device)

    has_tasks = (target_twists is not None and target_masks is not None and len(target_twists) > 0)
    num_modes = target_twists.shape[0] if has_tasks else 1

    for k in range(num_modes):
        # === 2. 求解一阶速度 q_dot (使用特征值分解) ===
        if has_tasks:
            # 构建增广矩阵 [J_loop; J_task]
            tgt_twist = target_twists[k]
            tgt_mask = target_masks[k].bool()

            if not tgt_mask.any():
                continue

            J_task_masked = J_path_0[tgt_mask, :]  # (D, N)
            target_masked = tgt_twist[tgt_mask].view(-1, 1)  # (D, 1)

            # [J_loop, 0]
            if J_loop_0.size(0) > 0:
                zeros_col = torch.zeros((J_loop_0.size(0), 1), device=device)
                top_block = torch.cat([J_loop_0, zeros_col], dim=1)
            else:
                top_block = torch.empty((0, num_nodes + 1), device=device)

            # [J_task, -target]
            bottom_block = torch.cat([J_task_masked, -target_masked], dim=1)
            M_aug = torch.cat([top_block, bottom_block], dim=0)

            # G = M^T * M
            G_aug = M_aug.T @ M_aug

            # [修复点 1] 使用 eigh 获取特征值和特征向量
            evals, evecs = torch.linalg.eigh(G_aug)

            # 取最小特征值对应的特征向量
            v_sol = evecs[:, 0]
            q_dot = v_sol[:num_nodes]

            # [梯度保护]
            q_norm = torch.norm(q_dot)
            q_dot = q_dot / (q_norm + 1e-8)

        else:
            # 无任务：仅解闭环零空间
            if J_loop_0.size(0) > 0:
                G_loop = J_loop_0.T @ J_loop_0
                # [修复点 2] 使用 eigh 获取特征值和特征向量
                evals, evecs = torch.linalg.eigh(G_loop)
                q_dot = evecs[:, 0]  # 最小特征值对应方向
            else:
                q_dot = torch.ones(num_nodes, device=device)
                q_dot = q_dot / torch.norm(q_dot)

        # 验证速度有效性
        v_ee_0 = J_path_0 @ q_dot
        v_norm_sq = torch.sum(v_ee_0 ** 2)

        # === 3. 计算二阶漂移 (Finite Difference) ===
        q_next = q_current + q_dot * dt
        J_loop_1, J_path_1, _ = _get_jacobians_and_screws(structure, q_next, loops, path_to_ee, base_node)

        # Drift = \dot{J} * \dot{q}
        drift_loop = (J_loop_1 @ q_dot - J_loop_0 @ q_dot) / dt

        # === 4. 求解二阶被动加速度 q_ddot (使用阻尼最小二乘法) ===
        if J_loop_0.size(0) > 0:
            damping = 1e-4
            JTJ = J_loop_0.T @ J_loop_0
            I = torch.eye(num_nodes, device=device)

            A_damped = JTJ + damping * I
            b_damped = J_loop_0.T @ (-drift_loop)

            try:
                q_ddot = torch.linalg.solve(A_damped, b_damped)
            except:
                q_ddot = torch.zeros_like(q_dot)
        else:
            q_ddot = torch.zeros_like(q_dot)

        # === 5. 计算末端全加速度 ===
        drift_ee = (J_path_1 @ q_dot - J_path_0 @ q_dot) / dt
        acc_ee = J_path_0 @ q_ddot + drift_ee

        # === 6. 正交投影判据 ===
        proj_scalar = torch.dot(acc_ee, v_ee_0) / (v_norm_sq + 1e-6)
        acc_parallel = proj_scalar * v_ee_0
        acc_perp = acc_ee - acc_parallel

        mode_loss = torch.norm(acc_perp) / (torch.norm(acc_ee) + 1e-6)
        total_loss += mode_loss

    if has_tasks:
        return total_loss / num_modes
    else:
        return total_loss
