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
    加载稀疏数据 (edge_list 6列 + joint_types 1D), 动态转换为 4 通道稠密张量, 并返回标签.
    4 通道: [mixed(exists/joint_type), a, alpha, offset]
    """

    def __init__(self, config, initial_manifest_path, augmented_manifest_path=None):
        self.manifest = []

        # --- 读取配置 ---
        self.config = config
        try:
            self.max_nodes = config['data']['max_nodes']
            self.num_features = config['diffusion_model']['in_channels']  # 应该是 4
            if self.num_features != 4:
                print(f"[警告] Dataloader 期望 in_channels=4, 但配置中为 {self.num_features}。")
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
                # 1. 加载稀疏数据
                #    edge_list_array: [num_edges, 6] (i, k, a, alpha, d_i, d_k)
                #    joint_types_array: [num_nodes] (0=P, 1=R)
                edge_list_array = data['edge_list_array']
                joint_types_array = data['joint_types_array']

                # 2. --- 动态转换为稠密张量 ---
                # 初始化一个全零的张量
                feature_tensor = np.zeros((self.max_nodes, self.max_nodes, self.num_features), dtype=np.float32)

                # 2a. 填充对角线 (Ch 0: 关节类型)
                #     (1.0 = R 副, 0.0 = P 副)
                for i, joint_type in enumerate(joint_types_array):
                    if i >= self.max_nodes: break
                    feature_tensor[i, i, 0] = joint_type

                # 2b. 遍历加载的边信息 (填充非对角线)
                for edge_data in edge_list_array:
                    # 解包 6 个值
                    i, k, a, alpha, d_i, d_k = edge_data
                    i, k = int(i), int(k)  # 确保是整数索引

                    if i >= self.max_nodes or k >= self.max_nodes:
                        continue

                    # 填充 4 个通道
                    # [mixed, a, alpha, offset]
                    feature_tensor[i, k] = [1.0, a, alpha, d_i]  # Ch 0 = 1.0 (exists)
                    feature_tensor[k, i] = [1.0, a, alpha, d_k]  # Ch 0 = 1.0 (exists)
                    # 确保 a 和 alpha 是对称的
                    feature_tensor[k, i, 1] = a
                    feature_tensor[k, i, 2] = alpha

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

        except KeyError as e:
            print(
                f"错误: 文件 {npz_path} 中未找到 'edge_list_array' 或 'joint_types_array'. 您是否使用了旧的数据格式? {e}")
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
    接收 *未归一化* 的 (H, W, 4) Numpy 张量列表,
    将其转换回 6 列 edge_list + 1D joint_types 格式进行保存。
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
        tensor_numpy = new_mech['tensor']  # (H, W, 4) 未归一化
        metadata = new_mech['metadata']

        # --- 核心转换: (H, W, 4) -> edge_list + joint_types ---
        edge_list_to_save = []
        max_nodes_tensor = tensor_numpy.shape[0]

        mixed_matrix = tensor_numpy[:, :, 0]  # Ch 0: Mixed (Exists / JointType)
        a_matrix = tensor_numpy[:, :, 1]  # Ch 1: a
        alpha_matrix = tensor_numpy[:, :, 2]  # Ch 2: alpha
        offset_matrix = tensor_numpy[:, :, 3]  # Ch 3: offset

        # 2a. 提取 1D 关节类型 (从对角线)
        joint_types_list = []
        for i in range(max_nodes_tensor):
            joint_type_val = mixed_matrix[i, i]
            # 1.0 = R 副, 0.0 = P 副 (根据您的要求)
            joint_types_list.append(1.0 if joint_type_val > 0.5 else 0.0)
        joint_types_array = np.array(joint_types_list, dtype=np.float32)

        # 2b. 提取 6 列 Edge List (从非对角线)
        for i in range(max_nodes_tensor):
            for k in range(i + 1, max_nodes_tensor):
                # 使用双向确认判断连接是否存在
                if mixed_matrix[i, k] > 0.5 and mixed_matrix[k, i] > 0.5:
                    a = a_matrix[i, k]
                    alpha = alpha_matrix[i, k]
                    d_i = offset_matrix[i, k]  # offset_ik
                    d_k = offset_matrix[k, i]  # offset_ki

                    edge_list_to_save.append(((i, k), a, alpha, d_i, d_k))

        if not edge_list_to_save:
            print("警告: 生成的张量解码后没有边, 跳过保存。")
            continue

        # --- 3. 保存稀疏数据 ---
        mech_id = f"mech_aug_{uuid.uuid4().hex[:12]}"
        npz_filename = f"{mech_id}.npz"

        filepath = os.path.join(base_dir, npz_filename)
        try:
            edge_save_array = []
            for edge_params in edge_list_to_save:
                (i, k), a, alpha, d_i, d_k = edge_params
                edge_save_array.append([
                    float(i), float(k),
                    float(a), float(alpha),
                    float(d_i), float(d_k)
                ])
            np.savez(
                filepath,
                edge_list_array=np.array(edge_save_array, dtype=np.float32),
                joint_types_array=joint_types_array
            )
        except Exception as e:
            print(f"错误: 无法保存文件 {filepath}. 原因: {e}")
            continue

        # --- 4. 创建 manifest 条目 ---
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