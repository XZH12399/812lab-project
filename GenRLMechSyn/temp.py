import numpy as np


def get_dh_matrix(theta, a, alpha, d):
    """
    根据标准DH参数计算变换矩阵 T_{i-1, i}
    Standard DH parameters:
    theta (q): 绕 z_{i-1} 旋转
    d (offset): 沿 z_{i-1} 移动
    a (length): 沿 x_i 移动 (公垂线长度)
    alpha: 绕 x_i 旋转 (扭转角)
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    T = np.array([
        [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
        [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0, s_alpha, c_alpha, d],
        [0, 0, 0, 1]
    ])
    return T

# 环路路径: [0, 1, 2, 3]
#     [Joint 0 | R] q= -1.2898  -->  [Link 0-1] a=  9.2207, alpha=  0.8711, d=  4.2901
#     [Joint 1 | R] q= -1.4797  -->  [Link 1-2] a=  8.3278, alpha=  0.7606, d=  4.8912
#     [Joint 2 | R] q= -1.2883  -->  [Link 2-3] a=  9.2727, alpha=  0.8701, d=  4.0808
#     [Joint 3 | R] q= -1.4791  -->  [Link 3-0] a=  8.2669, alpha=  0.7626, d=  4.8766


# 输入数据 (从 0 -> 1 -> 2 -> 3)
# 格式: [theta(q), a, alpha, d(offset)]
# 注意：只计算到Joint 2 -> Link 2-3，即获得Frame 3相对于Frame 0的位姿
dh_params = [
    # Joint 0 | Link 0-1
    {"q": -1.2898, "a": 9.2207, "alpha": 0.8711, "d": 4.2901},
    # Joint 1 | Link 1-2
    {"q": -1.4797, "a": 8.3278, "alpha": 0.7606, "d": 4.8912},
    # Joint 2 | Link 2-3
    {"q": -1.2883, "a": 9.2727, "alpha": 0.8701, "d": 4.0808},
    # # Joint 3 | Link 3-0
    # {"q": -1.4791, "a": 8.2669, "alpha": 0.7626, "d": 4.8766}
]

# 初始化单位矩阵
T_total = np.eye(4)

print("--- 分步计算变换矩阵 ---")
for i, param in enumerate(dh_params):
    # 获取当前步骤的变换矩阵
    T_i = get_dh_matrix(param["q"], param["a"], param["alpha"], param["d"])

    # 累乘矩阵 T_03 = T_01 * T_12 * T_23
    T_total = np.dot(T_total, T_i)

    print(f"Step {i} -> {i + 1}:")
    print(np.round(T_total, 4))
    print("-" * 30)

print("\n=== 最终结果: Frame 3 相对于 Frame 0 的位姿 (T_03) ===")
print(T_total)

# 提取位置和姿态信息
pos_x = T_total[0, 3]
pos_y = T_total[1, 3]
pos_z = T_total[2, 3]

print(f"\n末端位置 (Position):")
print(f"x = {pos_x:.4f}")
print(f"y = {pos_y:.4f}")
print(f"z = {pos_z:.4f}")

# 简单的旋转矩阵部分
R = T_total[:3, :3]
print("\n末端旋转矩阵 (Rotation Matrix):")
print(R)

# --- 验证环路闭合 (可选) ---
# 如果这是一个完美的闭环机构，乘以最后一个 Link 3-0 的变换矩阵后，应该回到单位矩阵(或接近原点)
link_3_0 = {"q": -1.4791, "a": 8.2669, "alpha": 0.7626, "d": 4.8766}
T_last = get_dh_matrix(link_3_0["q"], link_3_0["a"], link_3_0["alpha"], link_3_0["d"])
T_loop = np.dot(T_total, T_last)

print("\n=== 环路闭合验证 (0->1->2->3->0) ===")
print("如果闭环约束成立，位置误差应接近 0，旋转矩阵应接近单位矩阵")
print(T_loop)
print(f"闭环位置误差: x={T_loop[0, 3]:.4f}, y={T_loop[1, 3]:.4f}, z={T_loop[2, 3]:.4f}")