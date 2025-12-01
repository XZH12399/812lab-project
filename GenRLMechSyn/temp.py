import numpy as np
import re


def get_dh_matrix(theta, a, alpha, d):
    """
    根据标准DH参数计算变换矩阵 T_{i-1, i}
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


def parse_input_string(raw_text):
    """
    解析原始文本输入，自动提取关节类型(R/P), q, a, al(alpha), d 数值
    """
    params = []
    lines = raw_text.strip().split('\n')

    # --- 核心修改 1: 更新正则表达式 ---
    # 增加了 \[\d+\|([RP])\] 来捕获关节类型 (R或P)
    # 匹配示例: [3|P] q=1.3622 ...
    pattern = re.compile(r"\[\d+\|([RP])\].*?q=([-\d\.]+).*?a=([-\d\.]+).*?al=([-\d\.]+).*?d=([-\d\.]+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            # 提取数据
            j_type = match.group(1)  # 'R' or 'P'
            q_val = float(match.group(2))
            a_val = float(match.group(3))
            al_val = float(match.group(4))
            d_val = float(match.group(5))

            params.append({
                "type": j_type,
                "q": q_val,
                "a": a_val,
                "alpha": al_val,
                "d_log": d_val  # 注意：这里改名为 d_log，因为它的物理意义取决于类型
            })
    return params


# ==========================================
# 1. 原始文本 (包含 P 副)
# ==========================================
raw_input_data = """
--- Loop 1: [0, 4, 5, 6] (Kinematic) ---
    [0|P] q=-1.6224 --> a=5.7795, al=0.0195, d=-3.1755
    [4|P] q=-1.4668 --> a=13.9365, al=0.1241, d=0.0711
    [5|P] q=1.7686 --> a=2.6386, al=0.0038, d=3.0970
    [6|P] q=1.2447 --> a=17.0354, al=0.1408, d=-0.0004
"""

# ==========================================
# 2. 自动处理与计算
# ==========================================

dh_params = parse_input_string(raw_input_data)
print(f"成功解析 {len(dh_params)} 组 DH 参数。\n")

T_total = np.eye(4)

print(f"{'ID':<3} | {'Type':<4} | {'Theta (Input)':<13} | {'d (Input)':<13} | {'a':<10} | {'alpha':<10}")
print("-" * 70)

for i, param in enumerate(dh_params):
    # --- 核心修改 2: 根据类型分配 DH 参数 ---

    a_dh = param["a"]
    alpha_dh = param["alpha"]

    if param["type"] == 'R':
        # 转动副 (Revolute):
        # 变量是 theta (来自日志 q), d 是常数 (来自日志 d)
        theta_dh = param["q"]
        d_dh = param["d_log"]
    elif param["type"] == 'P':
        # 移动副 (Prismatic):
        # 变量是 d (来自日志 q), theta 是常数 (来自日志 d)
        theta_dh = param["d_log"]  # <--- 交换！
        d_dh = param["q"]  # <--- 交换！

    # 打印每一步实际使用的 DH 参数
    print(f"{i:<3} | {param['type']:<4} | {theta_dh:<13.4f} | {d_dh:<13.4f} | {a_dh:<10.4f} | {alpha_dh:<10.4f}")

    # 计算
    T_i = get_dh_matrix(theta_dh, a_dh, alpha_dh, d_dh)
    T_total = np.dot(T_total, T_i)

print("-" * 70)
print("\n=== 最终结果: 累积变换矩阵 (T_Total) ===")
with np.printoptions(precision=4, suppress=True):
    print(T_total)

# 提取位置
pos_x, pos_y, pos_z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
rot_trace = np.trace(T_total[:3, :3])

print(f"\n末端位置误差 (应该接近 0):")
print(f"x = {pos_x:.6f}")
print(f"y = {pos_y:.6f}")
print(f"z = {pos_z:.6f}")

# 闭环检测
# 位置误差容忍度
pos_error = np.sqrt(pos_x ** 2 + pos_y ** 2 + pos_z ** 2)
# 旋转误差检测 (单位阵的迹为3)
rot_error = abs(rot_trace - 3.0)

print("\n=== 闭环检测结果 ===")
if pos_error < 1e-3 and rot_error < 1e-3:
    print(f">> 这是一个完美的闭合回路！ (位置误差: {pos_error:.6f})")
else:
    print(f">> 回路未闭合 (位置误差: {pos_error:.6f})")
    print("   提示: 如果误差依然很大，请检查原始数据中 P 副的定义是否与假设一致。")