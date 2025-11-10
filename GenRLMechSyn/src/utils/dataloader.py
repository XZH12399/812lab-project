# src/utils/dataloader.py
import os
import json
import numpy as np
import torch
import uuid
from torch.utils.data import Dataset, DataLoader
import yaml  # <-- 导入 yaml 以读取 config


class MechanismDataset(Dataset):
    """
    加载稀疏 edge_list 数据 (8列), 动态转换为 5 通道稠密张量, 并返回标签.
    5 通道: [exists, joint_type, a, alpha, offset]
    """

    def __init__(self, config, initial_manifest_path, augmented_manifest_path=None):
        self.manifest = []

        # --- 读取配置 ---
        self.config = config
        try:
            self.max_nodes = config['data']['max_nodes']
            self.num_features = config['diffusion_model']['in_channels']  # 应该是 5
            if self.num_features != 5:
                print(f"[警告] Dataloader 期望 in_channels=5, 但配置中为 {self.num_features}。")
        except KeyError as e:
            raise ValueError(f"配置文件 config.yaml 缺少必要的键: {e}")

        # --- 创建标签到索引的映射 ---
        self.label_to_index = {"bennett": 0}

        # 1. 加载初始数据集
        self.initial_data_dir = os.path.dirname(initial_manifest_path)
        self.manifest.extend(self._load_manifest(initial_manifest_path, self.initial_data_dir))

        # 2. 加载增强数据集
        if augmented_manifest_path:
            self.augmented_data_dir = os.path.dirname(augmented_manifest_path)
            self.manifest.extend(self._load_manifest(augmented_manifest_path, self.augmented_data_dir))
        else:
            print("信息: 未提供增强数据集路径, 跳过加载。")

        print(
            f"--- 数据集加载完成: 共 {len(self.manifest)} 个机构 (将动态转换为 {self.max_nodes}x{self.max_nodes}x{self.num_features} 张量) ---")

    def _load_manifest(self, manifest_path, data_dir):
        if not os.path.exists(manifest_path):
            print(f"信息: Manifest 文件 {manifest_path} 不存在, 跳过加载。")
            return []
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_entries = json.load(f)
            loaded_entries = []
            for entry in manifest_entries:
                if 'metadata' in entry and 'label' in entry['metadata']:
                    entry['full_npz_path'] = os.path.join(data_dir, entry['data_path'])
                    loaded_entries.append(entry)
                else:
                    print(f"警告: 条目 {entry.get('id', 'N/A')} 缺少 metadata 或 label, 将被跳过。")

            print(f"从 {manifest_path} 成功加载 {len(loaded_entries)} 条有效元数据。")
            return loaded_entries
        except Exception as e:
            print(f"错误: 加载 manifest {manifest_path} 失败: {e}")
            return []

    def __len__(self):
        return len(self.manifest)

    # --- __getitem__ ---
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        npz_path = entry['full_npz_path']

        try:
            with np.load(npz_path) as data:
                # 1. 加载稀疏数据 (Numpy 数组: [num_edges, 8])
                #    列: [i, k, a, alpha, d_i, d_k, joint_type_i, joint_type_k]
                edge_list_array = data['edge_list_array']

                # 2. --- 动态转换为稠密张量 ---
                # 初始化一个全零的张量
                feature_tensor = np.zeros((self.max_nodes, self.max_nodes, self.num_features), dtype=np.float32)

                # 遍历加载的边信息
                for edge_data in edge_list_array:
                    # 解包 8 个值
                    i, k, a, alpha, d_i, d_k, joint_type_i, joint_type_k = edge_data
                    i, k = int(i), int(k)  # 确保是整数索引

                    if i >= self.max_nodes or k >= self.max_nodes:
                        continue

                    # 填充 5 个通道
                    # [exists, joint_type, a, alpha, offset]
                    feature_tensor[i, k] = [1.0, joint_type_i, a, alpha, d_i]
                    feature_tensor[k, i] = [1.0, joint_type_k, a, alpha, d_k]

                # 3. 转换为 PyTorch 张量 (C, H, W)
                # (H, W, C) -> (C, H, W)
                tensor_torch = torch.from_numpy(np.transpose(feature_tensor, (2, 0, 1))).float()

                # --- 获取标签并转换为索引 ---
                label_str = entry['metadata']['label']
                label_index = self.label_to_index.get(label_str, -1)
                if label_index == -1:
                    print(f"警告: 在 {npz_path} 中遇到未知标签 '{label_str}'")
                    label_index = 0

                    # --- 返回 (张量, 标签索引) ---
                return tensor_torch, torch.tensor(label_index, dtype=torch.long)

        except KeyError:
            print(f"错误: 文件 {npz_path} 中未找到 'edge_list_array'. 您是否使用了旧的数据格式?")
            return torch.zeros(self.num_features, self.max_nodes, self.max_nodes), torch.tensor(0, dtype=torch.long)
        except Exception as e:
            print(f"错误: 无法加载或转换文件 {npz_path}. 原因: {e}")
            # 返回一个空张量
            return torch.zeros(self.num_features, self.max_nodes, self.max_nodes), torch.tensor(0, dtype=torch.long)


def get_dataloader(config, initial_manifest_path, augmented_manifest_path, batch_size, shuffle=True):
    dataset = MechanismDataset(
        config,  # <-- 传递 config
        initial_manifest_path,
        augmented_manifest_path
    )
    if len(dataset) == 0:
        print("错误：数据集中没有任何数据！")
        return None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True
    )
    return loader


# --- 保存函数 ---
def add_mechanisms_to_dataset(mechanisms_to_add, manifest_path):
    """
    将一批新机构添加到数据集中.
    接收 *未归一化* 的 (H, W, 5) Numpy 张量列表,
    将其转换回 8 列的 edge_list 格式进行保存。
    """
    if not mechanisms_to_add:
        print("没有可添加的机构，跳过保存。")
        return

    base_dir = os.path.dirname(manifest_path)
    os.makedirs(base_dir, exist_ok=True)

    # 1. 加载现有 manifest
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: manifest 文件 {manifest_path} 格式错误, 将创建新的。")
            manifest = []
        except Exception as e:
            print(f"错误: 读取 manifest 文件 {manifest_path} 失败: {e}")
            manifest = []
    else:
        manifest = []

    # 2. 遍历新机构, 转换回 edge_list, 保存 NPZ, 创建 manifest 条目
    successful_saves = 0
    for new_mech in mechanisms_to_add:
        tensor_numpy = new_mech['tensor']  # (H, W, 5) 未归一化
        metadata = new_mech['metadata']

        # --- 核心转换: (H, W, 5) -> edge_list ---
        edge_list_to_save = []
        max_nodes_tensor = tensor_numpy.shape[0]
        exists_matrix = tensor_numpy[:, :, 0]
        joint_type_matrix = tensor_numpy[:, :, 1]
        a_matrix = tensor_numpy[:, :, 2]
        alpha_matrix = tensor_numpy[:, :, 3]
        offset_matrix = tensor_numpy[:, :, 4]

        for i in range(max_nodes_tensor):
            for k in range(i + 1, max_nodes_tensor):
                # 使用双向确认判断连接是否存在
                if exists_matrix[i, k] > 0.5 and exists_matrix[k, i] > 0.5:
                    a = a_matrix[i, k]
                    alpha = alpha_matrix[i, k]
                    d_i = offset_matrix[i, k]  # offset_ik
                    d_k = offset_matrix[k, i]  # offset_ki

                    # 获取关节类型
                    # joint_type_matrix[i, k] 存储 joint i 的类型
                    # joint_type_matrix[k, i] 存储 joint k 的类型
                    joint_type_i = joint_type_matrix[i, k]
                    joint_type_k = joint_type_matrix[k, i]

                    edge_list_to_save.append(((i, k), a, alpha, d_i, d_k, joint_type_i, joint_type_k))

        if not edge_list_to_save:
            print("警告: 生成的张量解码后没有边, 跳过保存。")
            continue

            # --- 保存稀疏数据 ---
        mech_id = f"mech_aug_{uuid.uuid4().hex[:12]}"
        npz_filename = f"{mech_id}.npz"

        filepath = os.path.join(base_dir, npz_filename)
        try:
            save_array = []
            # 解包 8 个值
            for edge_params in edge_list_to_save:
                (i, k), a, alpha, d_i, d_k, joint_type_i, joint_type_k = edge_params
                save_array.append([
                    float(i), float(k),
                    float(a), float(alpha),
                    float(d_i), float(d_k),
                    float(joint_type_i), float(joint_type_k)
                ])
            np.savez(filepath, edge_list_array=np.array(save_array, dtype=np.float32))
        except Exception as e:
            print(f"错误: 无法保存文件 {filepath}. 原因: {e}")
            continue

            # --- 创建 manifest 条目 ---
        new_entry = {"id": mech_id, "data_path": npz_filename, "metadata": metadata}
        manifest.append(new_entry)
        successful_saves += 1

    # 3. 写回 manifest 文件
    if successful_saves > 0:
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
            print(f"--- 数据集增强 (稀疏格式) ---")
            print(f"成功将 {successful_saves} 个新机构添加到 {manifest_path}。")
        except Exception as e:
            print(f"错误: 无法写回 manifest 文件 {manifest_path}. 原因: {e}")
    elif mechanisms_to_add:
        print("没有成功保存任何新机构。")
    else:
        print("没有可添加的机构，跳过保存。")