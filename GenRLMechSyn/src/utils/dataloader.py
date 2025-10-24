# src/utils/dataloader.py
import os
import json
import numpy as np
import torch
import uuid
from torch.utils.data import Dataset, DataLoader


class MechanismDataset(Dataset):
    """
    一个自定义的 PyTorch Dataset, 用于加载机构张量.
    它可以同时加载初始数据集和增强数据集。
    """

    def __init__(self, initial_manifest_path, augmented_manifest_path):
        self.manifest = []

        # 1. 加载初始数据集
        self.initial_data_dir = os.path.dirname(initial_manifest_path)
        self.manifest.extend(self._load_manifest(initial_manifest_path, self.initial_data_dir))

        # 2. 加载增强数据集 (如果提供了路径)
        if augmented_manifest_path:  # <-- 2. 添加 if 检查
            self.augmented_data_dir = os.path.dirname(augmented_manifest_path)
            self.manifest.extend(self._load_manifest(augmented_manifest_path, self.augmented_data_dir))
        else:
            print("信息: 未提供增强数据集路径, 跳过加载。")

        print(f"--- 数据集加载完成: 共 {len(self.manifest)} 个机构 ---")

    def _load_manifest(self, manifest_path, data_dir):
        if not os.path.exists(manifest_path):
            print(f"信息: Manifest 文件 {manifest_path} 不存在, 跳过加载。")
            return []

        with open(manifest_path, 'r') as f:
            manifest_entries = json.load(f)

        # 为每个条目添加完整的数据路径
        for entry in manifest_entries:
            entry['full_npz_path'] = os.path.join(data_dir, entry['data_path'])

        print(f"从 {manifest_path} 成功加载 {len(manifest_entries)} 条元数据。")
        return manifest_entries

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        npz_path = entry['full_npz_path']

        try:
            with np.load(npz_path) as data:
                # 这就是我们定义的核心张量 (30, 30, 4)
                tensor = data['mechanism_tensor']

                # 转换为 PyTorch 张量
                # 扩散模型(CNN/DiT)通常需要 (Channels, Height, Width)
                # 我们的 (30, 30, 4) 格式需要变为 (4, 30, 30)
                tensor = np.transpose(tensor, (2, 0, 1))
                return torch.from_numpy(tensor).float()

        except Exception as e:
            print(f"错误: 无法加载文件 {npz_path}. 原因: {e}")
            # 返回一个空张量, 以免训练崩溃
            return torch.zeros(4, 30, 30)


def get_dataloader(initial_manifest_path, augmented_manifest_path, batch_size, shuffle=True):
    """
    高级函数：创建并返回一个 PyTorch DataLoader.
    """
    dataset = MechanismDataset(
        initial_manifest_path,
        augmented_manifest_path # (这个可能是 None)
    )
    if len(dataset) == 0:
        print("错误：数据集中没有任何数据！")
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 简单起见, 暂时用0
        pin_memory=True
    )
    return loader


def add_mechanisms_to_dataset(mechanisms_to_add, manifest_path):
    """
    将一批新机构添加到数据集中 (保存NPZ并更新JSON)
    :param mechanisms_to_add: 包含元数据和张量的机构字典列表
    :param manifest_path: 要更新的 *augmented* manifest 文件路径
    """
    if not mechanisms_to_add:
        print("没有可添加的机构，跳过保存。")
        return

    base_dir = os.path.dirname(manifest_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 1. 加载现有的 manifest，如果不存在则创建一个空的
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = []

    # 2. 遍历新机构，保存 NPZ 文件并创建 manifest 条目
    for new_mech in mechanisms_to_add:
        # 创建唯一ID和文件名
        mech_id = f"mech_aug_{uuid.uuid4().hex[:12]}"
        npz_filename = f"{mech_id}.npz"
        npz_path = os.path.join(base_dir, npz_filename)

        # 保存数值数据 (mechanism_tensor)
        np.savez(npz_path, mechanism_tensor=new_mech['tensor'])

        # 创建新的 manifest 条目
        new_entry = {
            "id": mech_id,
            "data_path": npz_filename,  # 只保存相对路径
            "metadata": new_mech['metadata']
        }
        manifest.append(new_entry)

    # 3. 将更新后的 manifest 写回文件
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    print(f"--- 数据集增强 ---")
    print(f"成功将 {len(mechanisms_to_add)} 个新机构添加到 {manifest_path}。")