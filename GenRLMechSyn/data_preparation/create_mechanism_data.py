# data_preparation/create_mechanism_data.py
import numpy as np
import os
import math
import random
import json
from pathlib import Path

# --- 1. 全局配置 ---
# MAX_NODES 和 NUM_FEATURES 不再需要在这里定义

# --- (不变) Bennett 参数随机化范围 ---
MIN_LINK_LENGTH = 1.0
MAX_LINK_LENGTH = 20.0
MIN_ALPHA = 0.1
MAX_ALPHA = math.pi / 2.0 - 0.1
NUM_TO_GENERATE = 100

# --- 2. 路径设置 (不变) ---
script_path = Path(os.path.realpath(__file__))
script_dir = script_path.parent
project_root = script_dir.parent
output_dir = project_root / "data" / "initial_dataset"
output_dir.mkdir(parents=True, exist_ok=True)
metadata_filepath = output_dir / "metadata.json"


# --- 3. (核心修改!) 保存函数 ---
def save_mechanism_sparse(edge_list_data, filename):
    """(已更新!) 将 edge_list 保存到 .npz 文件."""
    filepath = output_dir / filename
    try:
        # --- 核心改动: 保存 edge_list ---
        # 我们需要将 list of tuples 转换为 Numpy object array 以便保存
        # edge_list_data 的格式: [((i, k), a, alpha, d_i, d_k), ...]
        # 我们将其转换为 [ [i, k, a, alpha, d_i, d_k], ...] 的 Numpy 数组
        save_array = []
        for edge_params in edge_list_data:
            (i, k), a, alpha, d_i, d_k = edge_params
            save_array.append([float(i), float(k), float(a), float(alpha), float(d_i), float(d_k)])

        np.savez(filepath, edge_list_array=np.array(save_array, dtype=np.float32))
        return True
    except Exception as e:
        print(f"错误: 无法保存文件 {filepath}. 原因: {e}")
        return False


# --- 4. (不变) Bennett 参数生成函数 ---
def generate_valid_bennett_params():
    """随机生成满足 Bennett 约束的参数 a1, a2, alpha1, alpha2"""
    # ... (此函数代码保持不变, 省略)
    while True:
        try:
            a1 = random.uniform(MIN_LINK_LENGTH, MAX_LINK_LENGTH)
            alpha1 = random.uniform(MIN_ALPHA, MAX_ALPHA)
            alpha2 = random.uniform(MIN_ALPHA, MAX_ALPHA)
            sin_alpha1 = math.sin(alpha1);
            sin_alpha2 = math.sin(alpha2)
            if abs(sin_alpha1) < 1e-6 or abs(sin_alpha2) < 1e-6: continue
            a2 = a1 * sin_alpha2 / sin_alpha1
            if MIN_LINK_LENGTH <= a2 <= MAX_LINK_LENGTH: return a1, a2, alpha1, alpha2
        except (ValueError, ZeroDivisionError):
            continue


# --- 5. 主执行逻辑 (已修改) ---
if __name__ == "__main__":
    print(f"开始随机生成 {NUM_TO_GENERATE} 个 Bennett 机构 (稀疏格式)...")
    print(f"数据将保存到: {output_dir}")

    all_metadata_entries = []
    successful_count = 0

    for i in range(NUM_TO_GENERATE):
        # 1. 生成有效的 Bennett 参数 (a 和 alpha)
        a1, a2, alpha1, alpha2 = generate_valid_bennett_params()

        # --- (核心修正!) 显式定义 d 值 ---
        # 对于 Bennett 机构, 所有连杆在相应轴上的偏移量 d 均为 0
        # 我们定义 d_ij 代表连杆 ij 在轴 i 上的偏移量
        d_01 = 0.0
        d_10 = 0.0
        d_12 = 0.0
        d_21 = 0.0
        d_23 = 0.0
        d_32 = 0.0
        d_30 = 0.0
        d_03 = 0.0
        # (请注意 d_ij 和 d_ji 通常是不同的物理量, 但在此特殊情况下都为0)
        # (将来生成其他机构时, 这些值可以不为零)

        # 2. 构造 edge_list (使用变量)
        # 格式: ( (i, k),    a_ik,  alpha_ik, d_i (on axis i), d_k (on axis k) )
        edge_list = [
            ((0, 1), a1, alpha1, d_01, d_10),  # 连杆 0<->1
            ((1, 2), a2, alpha2, d_12, d_21),  # 连杆 1<->2
            ((2, 3), a1, alpha1, d_23, d_32),  # 连杆 2<->3
            ((3, 0), a2, alpha2, d_30, d_03)  # 连杆 3<->0
        ]

        # edge_list = [
        #     ((0, 1), 0, 0, 0, 0),  # 连杆 0<->1
        #     ((1, 2), 0, 0, 0, 0),  # 连杆 1<->2
        #     ((2, 3), 0, 0, 0, 0),  # 连杆 2<->3
        #     ((3, 0), 0, 0, 0, 0)  # 连杆 3<->0
        # ]

        # 3. 生成文件名和 ID (不变)
        file_index = i + 1
        filename = f"Bennett_{file_index:03d}.npz"
        mechanism_id = f"bennett_{file_index:03d}"

        # 4. 构造元数据 (不变)
        metadata = {
            "source": "initial_random", "name": f"Random Bennett Mechanism ({file_index:03d})",
            "label": "bennett", "params": f"a={a1:.2f}/{a2:.2f}, alpha={alpha1:.3f}/{alpha2:.3f}"
        }

        # 5. 保存稀疏数据 (调用 save_mechanism_sparse, 不变)
        if save_mechanism_sparse(edge_list, filename):
            # 6. 准备 JSON 条目 (不变)
            json_entry = {"id": mechanism_id, "data_path": filename, "metadata": metadata}
            all_metadata_entries.append(json_entry)
            successful_count += 1
        else:
            print(f"跳过机构 {filename} 的元数据记录。")

        # 打印进度 (不变)
        if (i + 1) % 10 == 0: print(f"已处理 {i + 1}/{NUM_TO_GENERATE} 个机构...")

    # --- 6. 生成 metadata.json 文件 (不变) ---
    print(f"\n正在生成 metadata.json 文件...")
    try:
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_metadata_entries, f, indent=4, ensure_ascii=False)
        print(f"metadata.json 已成功保存至: {metadata_filepath}")
    except Exception as e:
        print(f"错误: 无法写入 metadata.json 文件. 原因: {e}")

    print(f"\n--- 完成 ---")
    print(f"总共成功创建了 {successful_count} / {NUM_TO_GENERATE} 个机构文件 (稀疏格式) 和对应的元数据。")