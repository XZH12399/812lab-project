# data_preparation/create_mechanism_data.py
import numpy as np
import os
from pathlib import Path

# --- 1. 全局配置 ---
MAX_NODES = 30
NUM_FEATURES = 4  # <-- 核心改动: 特征数改为 4 ([exists, a, alpha, d])

# --- 2. 路径设置 (保持不变) ---
script_path = Path(os.path.realpath(__file__))
script_dir = script_path.parent
project_root = script_dir.parent
output_dir = project_root / "data" / "initial_dataset"
output_dir.mkdir(parents=True, exist_ok=True)


# --- 3. 核心功能函数 ---

def create_mechanism_tensor(edge_list):
    """
    (已更新) 根据边列表生成一个 [MAX_NODES, MAX_NODES, 4] 的张量.
    通道 0: exists (1.0 or 0.0)
    通道 1: a
    通道 2: alpha
    通道 3: d_at_source
    """
    # 初始化一个全零的张量
    feature_tensor = np.zeros((MAX_NODES, MAX_NODES, NUM_FEATURES), dtype=np.float32)

    # 遍历列表中的每一个物理连杆
    for edge_params in edge_list:
        (i, k), a, alpha, d_i, d_k = edge_params

        if i >= MAX_NODES or k >= MAX_NODES:
            print(f"警告: 边 ({i}, {k}) 超出了 MAX_NODES={MAX_NODES} 的限制，将被忽略。")
            continue

        # --- 核心改动: 填充 4 个通道 ---
        # 填充 i -> k 的方向: [exists=1, a, alpha, d_i]
        feature_tensor[i, k] = [1.0, a, alpha, d_i]

        # 填充 k -> i 的方向: [exists=1, a, alpha, d_k]
        feature_tensor[k, i] = [1.0, a, alpha, d_k]

    # 其他位置默认为 [0.0, 0.0, 0.0, 0.0], 表示无连接

    return feature_tensor


def save_mechanism(tensor_data, filename):
    """(保持不变) 将机构张量保存到 .npz 文件."""
    filepath = output_dir / filename
    np.savez(filepath, mechanism_tensor=tensor_data)
    print(f"文件已成功保存至: {filepath}")


# --- 4. 定义您想生成的机构 (保持不变) ---
mechanisms_to_generate = [
    # --- 统一命名: bennett_xxx.npz ---
    {
        'filename': 'bennett_001.npz', # 原 valid_001
        'edge_list': [
            ( (0, 1), 10.0,    0.785, 0.0, 0.0 ), ( (1, 2), 14.14,   0.524, 0.0, 0.0 ),
            ( (2, 3), 10.0,    0.785, 0.0, 0.0 ), ( (3, 0), 14.14,   0.524, 0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 1)", "description": "Spatial: a=10/14.14, alpha=0.785/0.524"}
    },
    {
        'filename': 'bennett_002.npz', # 原 valid_002
        'edge_list': [
            ( (0, 1),  8.0,    1.0,   0.0, 0.0 ), ( (1, 2), 14.05,   0.5,   0.0, 0.0 ),
            ( (2, 3),  8.0,    1.0,   0.0, 0.0 ), ( (3, 0), 14.05,   0.5,   0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 2)", "description": "Spatial: a=8/14.05, alpha=1.0/0.5"}
    },
    {
        'filename': 'bennett_003.npz', # 原 valid_003
        'edge_list': [
            ( (0, 1), 24.74,   0.6,   0.0, 0.0 ), ( (1, 2), 15.0,    1.2,   0.0, 0.0 ),
            ( (2, 3), 24.74,   0.6,   0.0, 0.0 ), ( (3, 0), 15.0,    1.2,   0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 3)", "description": "Spatial: a=24.74/15, alpha=0.6/1.2"}
    },
    {
        'filename': 'bennett_004.npz', # 原 planar_001
        'edge_list': [
            ( (0, 1),  7.0,    0.0,   0.0, 0.0 ), ( (1, 2),  9.0,    0.0,   0.0, 0.0 ),
            ( (2, 3),  7.0,    0.0,   0.0, 0.0 ), ( (3, 0),  9.0,    0.0,   0.0, 0.0 )
        ], 'metadata': {"name": "Planar 4-Bar Mechanism (Set 1)", "description": "Planar: a=7/9, alpha=0.0"}
    },
    {
        'filename': 'bennett_005.npz', # 原 planar_002
        'edge_list': [
            ( (0, 1), 12.0,    0.0,   0.0, 0.0 ), ( (1, 2),  5.0,    0.0,   0.0, 0.0 ),
            ( (2, 3), 12.0,    0.0,   0.0, 0.0 ), ( (3, 0),  5.0,    0.0,   0.0, 0.0 )
        ], 'metadata': {"name": "Planar 4-Bar Mechanism (Set 2)", "description": "Planar: a=12/5, alpha=0.0"}
    },
    {
        'filename': 'bennett_006.npz', # 原 valid_004
        'edge_list': [
            ( (0, 1), 12.0,    1.1,   0.0, 0.0 ), ( (1, 2),  7.6,    0.6,   0.0, 0.0 ),
            ( (2, 3), 12.0,    1.1,   0.0, 0.0 ), ( (3, 0),  7.6,    0.6,   0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 4)", "description": "Spatial: a=12/7.6, alpha=1.1/0.6"}
    },
    {
        'filename': 'bennett_007.npz', # 原 valid_005
        'edge_list': [
            ( (0, 1),  5.0,    0.4,   0.0, 0.0 ), ( (1, 2), 10.06,   0.9,   0.0, 0.0 ),
            ( (2, 3),  5.0,    0.4,   0.0, 0.0 ), ( (3, 0), 10.06,   0.9,   0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 5)", "description": "Spatial: a=5/10.06, alpha=0.4/0.9"}
    },
    {
        'filename': 'bennett_008.npz', # 原 valid_006
        'edge_list': [
            ( (0, 1), 20.0,    0.2,   0.0, 0.0 ), ( (1, 2), 29.74,   0.3,   0.0, 0.0 ),
            ( (2, 3), 20.0,    0.2,   0.0, 0.0 ), ( (3, 0), 29.74,   0.3,   0.0, 0.0 )
        ], 'metadata': {"name": "Bennett Mechanism (Set 6)", "description": "Spatial: a=20/29.74, alpha=0.2/0.3"}
    },
    {
        'filename': 'bennett_009.npz', # 原 planar_003
        'edge_list': [
            ( (0, 1), 15.0,    0.0,   0.0, 0.0 ), ( (1, 2), 10.0,    0.0,   0.0, 0.0 ),
            ( (2, 3), 15.0,    0.0,   0.0, 0.0 ), ( (3, 0), 10.0,    0.0,   0.0, 0.0 )
        ], 'metadata': {"name": "Planar 4-Bar Mechanism (Set 3)", "description": "Planar: a=15/10, alpha=0.0"}
    },
    {
        'filename': 'bennett_010.npz', # 原 planar_004
        'edge_list': [
            ( (0, 1),  4.0,    0.0,   0.0, 0.0 ), ( (1, 2), 16.0,    0.0,   0.0, 0.0 ),
            ( (2, 3),  4.0,    0.0,   0.0, 0.0 ), ( (3, 0), 16.0,    0.0,   0.0, 0.0 )
        ], 'metadata': {"name": "Planar 4-Bar Mechanism (Set 4)", "description": "Planar: a=4/16, alpha=0.0"}
    }
]

# --- 5. 主执行逻辑 (保持不变) ---
if __name__ == "__main__":
    print(f"开始生成机构数据 (格式: [exists, a, alpha, d])，将保存到: {output_dir}")
    num_created = 0
    for mech_data in mechanisms_to_generate:
        try:
            tensor = create_mechanism_tensor(mech_data['edge_list'])
            save_mechanism(tensor, mech_data['filename'])
            num_created += 1
        except Exception as e:
            print(f"错误: 无法创建机构 {mech_data['filename']}. 原因: {e}")
    print(f"\n--- 完成 ---")
    print(f"总共成功创建了 {num_created} / {len(mechanisms_to_generate)} 个机构文件。")