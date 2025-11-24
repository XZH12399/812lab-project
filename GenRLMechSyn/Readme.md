# GenRLMechSyn: 项目介绍 & 技术路线图

**版本:** 1.2
**日期:** 2025年11月
**状态:** 生成 + 软约束优化 (Generation + Soft Optimization)

## 1. 项目愿景与目标

**GenRLMechSyn** (Generative Reinforcement Learning for Mechanism Synthesis) 项目旨在利用最前沿的生成式AI与物理仿真技术，彻底改变传统机械机构的设计范式。

**核心变革:** 我们从纯粹的“生成即所得”模式，进化为**“生成-精修-学习” (Generate-Refine-Learn)** 模式。系统不仅能“画”出机构，还能利用可微物理引擎自动“修”好机构，最终通过强化学习让生成器学会直接生成高质量的机构。

**核心目标:** 构建一个闭环系统，结合 **Diffusion Transformer (DiT)** 的创造力、**可微运动学 (Differentiable Kinematics)** 的优化能力以及 **强化学习 (RL)** 的引导能力，实现复杂空间机构（如 Bennett 机构）的自动化综合。

## 2. 核心架构：生成-精修-评估 (Generate-Refine-Evaluate)

本项目采用了深度学习与传统优化相结合的混合架构：

* **1. 模仿者 (The Mimic): DiT (Diffusion Transformer)**
    * **角色:** “建筑师的草图绘制员”。
    * **功能:** 负责在巨大的设计空间中进行探索，输出包含拓扑结构和粗糙几何参数的“机构初稿”。它不需要一步到位生成完美的机构，只需提供一个良好的优化起点。
    * **数据格式:** 5通道张量 `(N, N, 5)`，包含 `[exists, joint_type, a, alpha, offset]`。

* **2. 精修者 (The Refiner): Soft Constraint Optimizer**
    * **角色:** “结构工程师” (v1.2 新增核心)。
    * **技术:** 基于 PyTorch 的**投影梯度下降 (Projected Gradient Descent)**。
    * **功能:** 接收 DiT 的初稿，在保持拓扑不变的前提下，微调几何参数 (`a, alpha, offset, q`)。
    * **物理引擎:** 内置可微的正向运动学求解器 (`simple_kinematics.py`)，计算闭环误差和几何特征误差。
    * **关键策略:**
        * **投影梯度:** 在每一步更新后，强制将参数投影回物理合法范围（如杆长非负、角度周期性），确保优化器与评估器视角一致。
        * **防退化约束:** 引入“最小杆长惩罚”，防止机构退化为三角形（3杆）或点。

* **3. 探索者与裁判 (The Explorer & Judge): RL Agent + Evaluator**
    * **角色:** “验收官”与“导师”。
    * **Evaluator (裁判):** 使用严格的“硬指标”对精修后的机构进行打分（如：必须是4杆、必须全R副、闭环误差<1e-3、满足Bennett几何比例）。
    * **RL Agent (导师):** 学习预测精修后机构的得分，并反向引导 DiT，使其下一次生成的初稿更容易被“修”成高分机构。

## 3. 系统工作流 (Workflow)

系统通过以下循环进行迭代进化：

1.  **引导采样 (Guided Sampling)**:
    DiT 在 RL 的引导下，生成一批机构的**原始张量 (Raw Tensor)**。此时的机构可能无法闭环，参数可能粗糙。

2.  **拓扑筛选 (Topology Filter)**:
    将张量转换为图 (Graph)，剔除无环结构或**退化结构**（如节点数 < 4 的三角形）。

3.  **软约束优化 (Soft Optimization)**:
    对筛选出的机构进行 **50步** 的梯度下降优化：
    * **变量**: 同时优化结构参数 ($S$) 和关节变量 ($Q$)。
    * **Loss**: `闭环误差 (相对)` + `Bennett几何误差` + `边界惩罚` + `正则化`。
    * **约束**: 强制杆长 $a > 0.5$ (物理单位)，强制关节类型为 R 副。
    * **输出**: 得到优化后的最优参数 `best_x` 和 `best_q`。

4.  **严格评估 (Strict Evaluation)**:
    Evaluator 接收优化后的参数，进行分级检查：
    * Level 1: 拓扑是否为 4-Cycle？
    * Level 2: 关节是否全为 R 副？
    * Level 3: 偏移量 (offset) 是否接近 0？
    * Level 4: 几何是否对称 ($a_1=a_3, \alpha_1=\alpha_3$)？
    * Level 5: 最终得分计算。

5.  **学习与增强 (Learn & Augment)**:
    * **RL更新**: 使用 `(原始初稿, 最终得分)` 更新 RL Agent。
    * **数据增强**: 将**优化后且高分**的机构加入数据集，DiT 在下一轮训练中学习这些“精修后的完美样本”。

## 4. 关键技术细节 (v1.2 更新)

### 4.1 投影梯度下降 (Projected Gradient Descent)
为了解决优化器（看到负数也能闭环）与评估器（截断负数为0导致闭环破坏）之间的**认知偏差 (Cognitive Gap)**，我们实施了严格的投影策略：
* **优化时**: 使用 `abs(a)` 和 `alpha % 2pi` 保持梯度流，允许中间值越界。
* **更新后**: 立即执行 `clamp_`，强制 $a_{norm} > -0.95$ (即物理值 $>0.5$)。
* **效果**: 保证了 Evaluation 阶段看到的参数与 Optimization 阶段计算 Loss 的参数在数学上完全等价。

### 4.2 相对误差指标 (Relative Error Metrics)
废弃了绝对误差，全面采用**相对误差**：
* **闭环误差**: $\text{PosError} / (\text{MaxLinkLength})^2$
* **几何误差**: $|x-y| / (|x|+|y|+\epsilon)$
* 这消除了机构尺寸对优化的影响，防止优化器通过“缩小机构”来作弊。

### 4.3 防退化机制 (Anti-Degeneration)
针对早期版本中机构倾向于退化为“三角形”（将一条边缩为0）的问题：
* **初始化**: 强制初始化杆长为较大值。
* **Loss惩罚**: 引入高权重的 `ReLU(0.5 - a)` 惩罚。
* **硬约束**: 投影阶段强制 $a \ge 0.5$。

## 5. 文件说明

* **`src/pipeline.py`**:
    * 核心训练循环。包含了从 DiT 采样到 Optimizer 优化，再到 Evaluator 评分的全过程。
* **`src/solver/simple_kinematics.py`**:
    * **新增**: 可微的正向运动学求解器。包含 `compute_loop_errors` (闭环) 和 `compute_bennett_geometry_error` (Bennett几何特征)。
    * 实现了 `abs` 和 `modulo` 运算以支持无截断的梯度传播。
* **`src/evaluator/evaluator.py`**:
    * **升级**: 支持显式传入 `known_loops`（解决方向歧义）。
    * 实现了分级硬门限检查 (Hard Thresholding)。
* **`configs/default_config.yaml`**:
    * 新增 `optimization_tolerance`: 控制进入评估阶段的 Loss 门槛。

## 6. 快速开始

1.  **环境配置**: Python 3.8+, PyTorch 2.0+ (建议使用 GPU)。
2.  **生成数据**: `python data_preparation/create_mechanism_data.py`。
3.  **开始训练**: `python scripts/train.py`
    * 观察控制台输出的 `[Debug Mech 0]` 日志，确认 `Closure` 误差在 50 步内显著下降。
    * 等待 `Mech XX: OK` 出现，并查看 `[详细参数报告]`。

## 7. 当前挑战

* **Bennett 几何的极难收敛**: 虽然闭环容易达成，但在软约束下同时满足 $a/\sin\alpha$ 比例和对称性仍具挑战，需要精心调节 Loss 权重。
* **RL 引导的滞后性**: RL 需要在 Replay Buffer 积累足够多的正样本后才能发挥显著作用。