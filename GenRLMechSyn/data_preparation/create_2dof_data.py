import numpy as np
import os
import uuid
import json
import random
from pathlib import Path

# ==========================================
# 1. 路径与全局配置
# ==========================================
# 动态获取路径 (参考 create_mechanism_data.py)
script_path = Path(os.path.realpath(__file__))
script_dir = script_path.parent
project_root = script_dir.parent

# 输出目录设置
output_dir = project_root / "data" / "initial_2dof_dataset"
output_dir.mkdir(parents=True, exist_ok=True)
metadata_filepath = output_dir / "metadata.json"

# 生成参数配置
NUM_SAMPLES = 100
MIN_LINK_LENGTH = 2.0
MAX_LINK_LENGTH = 15.0
OFFSET_RANGE = 5.0

# ==========================================
# 2. 种子拓扑定义 (8-Bar Asymmetric Loop)
# ==========================================
# 目标: 构建一个空间 2-DOF 机构 (等效于双分支并联)
# 节点: 8个 (0 到 7)
# 约定:
#   - Node 0: 基座 (Base)
#   - Node 7: 末端平台入口 (EE Entry Joint)
#
# 拓扑设计 (为了让 shortest_path 稳定选中分支 1):
#   - 分支 1 (短链): 0 -> 1 -> 2 -> 7  (长度 3跳) -> 用于计算末端运动
#   - 分支 2 (长链): 0 -> 3 -> 4 -> 5 -> 6 -> 7 (长度 5跳) -> 提供闭环约束

SEED_TOPOLOGY = {
    "num_nodes": 8,
    "edges": [
        # --- Branch 1 (主要运动链) ---
        (0, 1), (1, 2), (2, 7),

        # --- Branch 2 (闭环约束链) ---
        (0, 3), (3, 4), (4, 5), (5, 6), (6, 7)
    ],
    "meta": {
        "topology_name": "8-bar-asymmetric-loop",
        "base_node": 0,
        "ee_node": 7
    }
}


def generate_random_mechanism(topology_def, sample_idx):
    """
    基于给定拓扑，随机生成几何参数。
    返回格式: [i, k, a, alpha, d_i, d_k, type_i, type_k]
    """
    edge_list_data = []

    for (u, v) in topology_def["edges"]:
        # A. 几何参数
        a = random.uniform(MIN_LINK_LENGTH, MAX_LINK_LENGTH)
        alpha = random.uniform(-np.pi, np.pi)
        d_u = random.uniform(-OFFSET_RANGE, OFFSET_RANGE)
        d_v = random.uniform(-OFFSET_RANGE, OFFSET_RANGE)

        # B. 关节类型
        # 80% R副, 20% P副 (由优化器进一步筛选)
        type_u = 1.0 if random.random() > 0.2 else -1.0
        type_v = 1.0 if random.random() > 0.2 else -1.0

        edge_list_data.append([
            float(u), float(v),
            a, alpha,
            d_u, d_v,
            type_u, type_v
        ])

    return np.array(edge_list_data, dtype=np.float32)


# ==========================================
# 3. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    metadata_list = []

    print(f"--- 开始生成 8-Bar 2-DOF 数据集 ---")
    print(f"目标数量: {NUM_SAMPLES}")
    print(f"保存路径: {output_dir}")
    print(f"拓扑结构: 0 -> 1 -> 2 -> 7 (EE)")

    for i in range(NUM_SAMPLES):
        # 1. 生成数据
        edge_array = generate_random_mechanism(SEED_TOPOLOGY, i)

        # 2. 生成文件名
        mech_id = f"2dof_8bar_{uuid.uuid4().hex[:8]}"
        filename = f"{mech_id}.npz"
        full_path = output_dir / filename

        try:
            # 3. 保存 .npz
            np.savez(full_path, edge_list_array=edge_array)

            # 4. 记录元数据
            meta_entry = {
                "id": mech_id,
                "data_path": filename,
                "metadata": {
                    "label": "general_2dof",
                    "source": "random_augmentation",
                    "topology": SEED_TOPOLOGY["meta"]["topology_name"],
                    "base_node": 0,
                    "ee_node": 7,  # 明确指向最大节点
                    "num_nodes": 8
                }
            }
            metadata_list.append(meta_entry)
        except Exception as e:
            print(f"保存失败 {i}: {e}")

        if (i + 1) % 200 == 0:
            print(f"已生成 {i + 1}/{NUM_SAMPLES} ...")

    # 5. 保存 metadata.json
    try:
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        print(f"--- 完成! 元数据已保存至 {metadata_filepath} ---")
    except Exception as e:
        print(f"错误: 无法写入 metadata.json. 原因: {e}")