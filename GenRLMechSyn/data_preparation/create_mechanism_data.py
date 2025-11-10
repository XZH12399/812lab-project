# data_preparation/create_mechanism_data.py
import numpy as np
import os
import math
import random
import json
from pathlib import Path

# --- 1. 全局配置 ---
MIN_LINK_LENGTH = 1.0
MAX_LINK_LENGTH = 20.0
MIN_ALPHA = 0.1
MAX_ALPHA = math.pi / 2.0 - 0.1
NUM_TO_GENERATE = 100

# --- 2. 路径设置 ---
script_path = Path(os.path.realpath(__file__))
script_dir = script_path.parent
project_root = script_dir.parent
output_dir = project_root / "data" / "initial_dataset"
output_dir.mkdir(parents=True, exist_ok=True)
metadata_filepath = output_dir / "metadata.json"


# --- 3. 保存函数 ---
def save_mechanism_sparse(edge_list_data, joint_types, filename):
    """
    将 edge_list 和 joint_types 保存到 .npz 文件.
    保存 8 列: [i, k, a, alpha, d_i, d_k, joint_type_i, joint_type_k]
    """
    filepath = output_dir / filename
    try:
        # --- 保存 edge_list 和 joint_types ---
        save_array = []
        for edge_params in edge_list_data:
            (i, k), a, alpha, d_i, d_k = edge_params
            i_int, k_int = int(i), int(k)  # 用于字典查找

            # (i, k, a, alpha, d_i, d_k, joint_type_i, joint_type_k)
            save_array.append([
                float(i), float(k),
                float(a), float(alpha),
                float(d_i), float(d_k),
                joint_types[i_int], joint_types[k_int]  # <-- 添加关节类型
            ])

        np.savez(filepath, edge_list_array=np.array(save_array, dtype=np.float32))
        return True
    except Exception as e:
        print(f"错误: 无法保存文件 {filepath}. 原因: {e}")
        return False


# --- 4. Bennett 参数生成函数 ---
def generate_valid_bennett_params():
    """随机生成满足 Bennett 约束的参数 a1, a2, alpha1, alpha2"""
    while True:
        try:
            a1 = random.uniform(MIN_LINK_LENGTH, MAX_LINK_LENGTH)
            alpha1 = random.uniform(MIN_ALPHA, MAX_ALPHA)
            alpha2 = random.uniform(MIN_ALPHA, MAX_ALPHA)
            sin_alpha1 = math.sin(alpha1);
            sin_alpha2 = math.sin(alpha2)
            if abs(sin_alpha1) < 1e-6 or abs(sin_alpha2) < 1e-6: continue
            a2 = a1 * sin_alpha1 / sin_alpha2
            if MIN_LINK_LENGTH <= a2 <= MAX_LINK_LENGTH: return a1, a2, alpha1, alpha2
        except (ValueError, ZeroDivisionError):
            continue


# --- 5. 主执行逻辑 ---
if __name__ == "__main__":
    print(f"开始随机生成 {NUM_TO_GENERATE} 个 Bennett 机构 (稀疏格式)...")
    print(f"数据将保存到: {output_dir}")

    all_metadata_entries = []
    successful_count = 0

    for i in range(NUM_TO_GENERATE):
        # 1. 生成有效的 Bennett 参数 (a 和 alpha)
        a1, a2, alpha1, alpha2 = generate_valid_bennett_params()

        # --- 显式定义 d (offset) 和 joint_type ---
        # 对于 Bennett 机构, 所有偏移量 (offset) 均为 0
        d_01, d_10 = 0.0, 0.0
        d_12, d_21 = 0.0, 0.0
        d_23, d_32 = 0.0, 0.0
        d_30, d_03 = 0.0, 0.0

        # 定义关节类型: Bennett 都是 R 副 (+1.0)
        joint_types = {
            0: 1.0,  # 关节 0 是 R 副
            1: 1.0,  # 关节 1 是 R 副
            2: 1.0,  # 关节 2 是 R 副
            3: 1.0  # 关节 3 是 R 副
        }
        # (将来如果生成 P 副, 这里的值应为 -1.0)

        # 2. 构造 edge_list
        # 格式: ( (i, k),    a_ik,  alpha_ik, d_i (on axis i), d_k (on axis k) )
        edge_list = [
            ((0, 1), a1, alpha1, d_01, d_10),  # 连杆 0<->1
            ((1, 2), a2, alpha2, d_12, d_21),  # 连杆 1<->2
            ((2, 3), a1, alpha1, d_23, d_32),  # 连杆 2<->3
            ((3, 0), a2, alpha2, d_30, d_03)  # 连杆 3<->0
        ]

        # 3. 生成文件名和 ID
        file_index = i + 1
        filename = f"Bennett_{file_index:03d}.npz"
        mechanism_id = f"bennett_{file_index:03d}"

        # 4. 构造元数据
        metadata = {
            "source": "initial_random", "name": f"Random Bennett Mechanism ({file_index:03d})",
            "label": "bennett", "params": f"a={a1:.2f}/{a2:.2f}, alpha={alpha1:.3f}/{alpha2:.3f}"
        }

        # 5. 保存稀疏数据 (调用 save_mechanism_sparse, 传入 joint_types)
        if save_mechanism_sparse(edge_list, joint_types, filename):
            # 6. 准备 JSON 条目
            json_entry = {"id": mechanism_id, "data_path": filename, "metadata": metadata}
            all_metadata_entries.append(json_entry)
            successful_count += 1
        else:
            print(f"跳过机构 {filename} 的元数据记录。")

        # 打印进度
        if (i + 1) % 10 == 0: print(f"已处理 {i + 1}/{NUM_TO_GENERATE} 个机构...")

    # --- 6. 生成 metadata.json 文件 ---
    print(f"\n正在生成 metadata.json 文件...")
    try:
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_metadata_entries, f, indent=4, ensure_ascii=False)
        print(f"metadata.json 已成功保存至: {metadata_filepath}")
    except Exception as e:
        print(f"错误: 无法写入 metadata.json 文件. 原因: {e}")

    print(f"\n--- 完成 ---")
    print(f"总共成功创建了 {successful_count} / {NUM_TO_GENERATE} 个机构文件 (稀疏格式) 和对应的元数据。")