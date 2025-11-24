# src/solver/simple_kinematics.py
import torch
import math
from .utils import get_dh_matrix


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
