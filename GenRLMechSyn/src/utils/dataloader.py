# src/utils/dataloader.py
import os
import json
import numpy as np
import torch
import uuid
from torch.utils.data import Dataset, DataLoader
import yaml


class MechanismDataset(Dataset):
    """
    加载机构数据集。
    (v1.2 修正: 从配置文件读取 label_mapping，支持多种机构类型)
    """

    def __init__(self, config, initial_manifest_path, augmented_manifest_path=None):
        self.manifest = []
        self.config = config

        # --- 1. 读取配置 ---
        try:
            self.max_nodes = config['data']['max_nodes']
            self.num_features = config['diffusion_model']['in_channels']  # 应该是 5

            # [核心修正] 从配置文件读取标签映射
            # 以前是写死的 {"bennett": 0}，现在改为动态读取
            # 如果配置文件里没写，默认只支持 bennett
            self.label_to_index = config['data'].get('label_mapping', {"bennett": 0})

            if self.num_features != 5:
                print(f"[警告] Dataloader 期望 in_channels=5, 但配置中为 {self.num_features}。")
        except KeyError as e:
            raise ValueError(f"配置文件 config.yaml 缺少必要的键: {e}")

        # --- 2. 加载数据清单 ---
        self.initial_data_dir = os.path.dirname(initial_manifest_path)
        self.manifest.extend(self._load_manifest(initial_manifest_path, self.initial_data_dir))

        if augmented_manifest_path:
            self.augmented_data_dir = os.path.dirname(augmented_manifest_path)
            self.manifest.extend(self._load_manifest(augmented_manifest_path, self.augmented_data_dir))
        else:
            print("信息: 未提供增强数据集路径, 跳过加载。")

        print(f"--- 数据集加载完成: 共 {len(self.manifest)} 个机构 (支持标签: {list(self.label_to_index.keys())}) ---")

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

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        npz_path = entry['full_npz_path']

        try:
            with np.load(npz_path) as data:
                # 1. 加载稀疏数据
                edge_list_array = data['edge_list_array']

                # 2. 转换为稠密张量
                feature_tensor = np.zeros((self.max_nodes, self.max_nodes, self.num_features), dtype=np.float32)
                joint_types_filled = set()

                for edge_data in edge_list_array:
                    # 兼容旧格式 (6列) 和新格式 (8列)
                    if len(edge_data) == 8:
                        i, k, a, alpha, d_i, d_k, joint_type_i, joint_type_k = edge_data
                    elif len(edge_data) == 6:
                        i, k, a, alpha, d_i, d_k = edge_data
                        joint_type_i, joint_type_k = 1.0, 1.0
                    else:
                        continue

                    i, k = int(i), int(k)
                    if i >= self.max_nodes or k >= self.max_nodes: continue

                    # 填充边属性
                    feature_tensor[i, k, 0] = 1.0
                    feature_tensor[i, k, 2] = a
                    feature_tensor[i, k, 3] = alpha
                    feature_tensor[i, k, 4] = d_i

                    feature_tensor[k, i, 0] = 1.0
                    feature_tensor[k, i, 2] = a
                    feature_tensor[k, i, 3] = alpha
                    feature_tensor[k, i, 4] = d_k

                    # 填充节点属性 (joint_type)
                    if i not in joint_types_filled:
                        feature_tensor[i, :, 1] = joint_type_i
                        joint_types_filled.add(i)
                    if k not in joint_types_filled:
                        feature_tensor[k, :, 1] = joint_type_k
                        joint_types_filled.add(k)

                tensor_torch = torch.from_numpy(np.transpose(feature_tensor, (2, 0, 1))).float()

                # --- 3. 获取标签并转换为索引 ---
                label_str = entry['metadata']['label']

                # [核心修正] 使用从 config 读取的映射表
                label_index = self.label_to_index.get(label_str, -1)

                if label_index == -1:
                    # 仅当标签真的不在 config 中时才警告
                    print(f"警告: 在 {npz_path} 中遇到未知标签 '{label_str}' (Config中未定义)")
                    label_index = 0

                return tensor_torch, torch.tensor(label_index, dtype=torch.long)

        except Exception as e:
            print(f"错误: 无法加载文件 {npz_path}. 原因: {e}")
            return torch.zeros(self.num_features, self.max_nodes, self.max_nodes), torch.tensor(0, dtype=torch.long)


def get_dataloader(config, initial_manifest_path, augmented_manifest_path, batch_size, shuffle=True):
    dataset = MechanismDataset(config, initial_manifest_path, augmented_manifest_path)
    if len(dataset) == 0:
        print("错误：数据集中没有任何数据！")
        return None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True
    )
    return loader


def add_mechanisms_to_dataset(mechanisms_to_add, manifest_path):
    # (保持不变)
    if not mechanisms_to_add: return

    base_dir = os.path.dirname(manifest_path)
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except:
            manifest = []
    else:
        manifest = []

    successful_saves = 0
    for new_mech in mechanisms_to_add:
        tensor_numpy = new_mech['tensor']
        metadata = new_mech['metadata']
        edge_list_to_save = []
        max_nodes_tensor = tensor_numpy.shape[0]

        # 提取矩阵
        exists = tensor_numpy[:, :, 0]
        j_type = tensor_numpy[:, :, 1]
        a_mat = tensor_numpy[:, :, 2]
        alp_mat = tensor_numpy[:, :, 3]
        off_mat = tensor_numpy[:, :, 4]

        for i in range(max_nodes_tensor):
            for k in range(i + 1, max_nodes_tensor):
                if exists[i, k] > 0.5:
                    edge_list_to_save.append([
                        i, k, a_mat[i, k], alp_mat[i, k], off_mat[i, k], off_mat[k, i],
                        j_type[i, 0], j_type[k, 0]
                    ])

        if not edge_list_to_save: continue

        mech_id = f"mech_aug_{uuid.uuid4().hex[:12]}"
        npz_filename = f"{mech_id}.npz"
        filepath = os.path.join(base_dir, npz_filename)
        try:
            np.savez(filepath, edge_list_array=np.array(edge_list_to_save, dtype=np.float32))
            manifest.append({"id": mech_id, "data_path": npz_filename, "metadata": metadata})
            successful_saves += 1
        except:
            continue

    if successful_saves > 0:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print(f"成功将 {successful_saves} 个新机构添加到 {manifest_path}。")