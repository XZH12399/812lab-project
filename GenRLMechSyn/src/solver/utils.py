# src/solver/utils.py
import torch
import networkx as nx
import numpy as np


def get_dh_matrix(a, alpha, d, theta):
    """
    构建标准 DH 变换矩阵 (支持 PyTorch 梯度)
    T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
    """
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    # 构造 4x4 矩阵的各行
    # Row 1: [ct, -st*ca, st*sa, a*ct]
    r1 = torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1)
    # Row 2: [st, ct*ca, -ct*sa, a*st]
    r2 = torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1)
    # Row 3: [0, sa, ca, d]
    zero = torch.zeros_like(ct)
    r3 = torch.stack([zero, sa, ca, d], dim=-1)
    # Row 4: [0, 0, 0, 1]
    one = torch.ones_like(ct)
    r4 = torch.stack([zero, zero, zero, one], dim=-1)

    T = torch.stack([r1, r2, r3, r4], dim=-2)  # (..., 4, 4)
    return T


def tensor_to_graph(structure_tensor):
    """
    将 (N, N, 5) 张量转换为 NetworkX 图。
    (已修正) 先对称化 exists 通道，再进行阈值判断。
    """
    if torch.is_tensor(structure_tensor):
        mat = structure_tensor.detach().cpu().numpy()
    else:
        mat = structure_tensor

    if mat.shape[0] == 5:
        mat = np.transpose(mat, (1, 2, 0))

    N = mat.shape[0]
    exists = mat[:, :, 0]

    # --- [核心修改] 先强制对称化 ---
    # (M + M.T) / 2
    exists_sym = (exists + exists.T) / 2.0

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # 基于对称后的矩阵建立连接
    rows, cols = np.where(exists_sym > 0.5)
    for r, c in zip(rows, cols):
        if r < c:
            G.add_edge(r, c)

    return G


def find_independent_loops(G):
    """
    寻找基本回路 (Fundamental Cycles)
    """
    try:
        cycles = nx.cycle_basis(G)
        return cycles
    except Exception:
        return []