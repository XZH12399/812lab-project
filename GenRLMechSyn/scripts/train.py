# scripts/train.py

import argparse
import yaml
import os
import sys
import traceback
import logging
import time
from datetime import datetime
import random  # <-- 导入 random
import numpy as np # <-- 导入 numpy
import torch   # <-- 导入 torch

# --- 关键的路径设置 ---
# 这段代码确保无论您从哪里运行这个脚本,
# Python 都能找到 'src' 文件夹。

# 1. 获取此脚本 (train.py) 所在的目录 (e.g., /path/to/GenRLMechSyn/scripts)
script_dir = os.path.dirname(os.path.realpath(__file__))

# 2. 获取项目根目录 (e.g., /path/to/GenRLMechSyn)
project_root = os.path.dirname(script_dir)

# 3. 将项目根目录添加到 Python 的模块搜索路径中
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"已将项目根目录添加到 Python 路径: {project_root}")

# --- 设置随机数种子的函数 ---
def set_seed(seed):
    """固定所有相关的随机数种子以确保可复现性"""
    logger = logging.getLogger()  # 获取 logger
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 为所有 GPU 设置种子
    logger.info(f"--- 所有随机数种子已固定为: {seed} ---")

    # --- (可选但推荐) 设置 Pytorch 的确定性行为 ---
    # 注意: 这可能会显著降低训练速度, 尤其是在使用卷积时
    # 并且并非所有操作都能保证完全确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    # 如果遇到某些操作仍然不确定, 可能需要:
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # 或者 ':16:8'
    # torch.use_deterministic_algorithms(True) # PyTorch 1.8+
    logger.info("--- PyTorch CUDNN 已设置为确定性模式 (可能影响性能) ---")

# --- 配置日志记录 ---
def setup_logging(log_dir="logs", run_name=None):
    """配置 logging 模块以输出到控制台和文件"""
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_filename = os.path.join(log_dir, f"{run_name}.log")
    os.makedirs(log_dir, exist_ok=True) # 确保目录存在

    # 获取根 logger
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # 设置最低日志级别

    # 移除 Pytorch 可能添加的默认处理器，避免重复输出
    # (或者直接获取 'GenRLMechSyn' logger)
    # for handler in logger.handlers[:]:
    #    logger.removeHandler(handler)

    # 创建文件处理器 (写入日志文件)
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器 (输出到屏幕)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # 控制台只输出消息
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    print(f"--- 日志记录已启动 ---")
    print(f"日志文件: {log_filename}")

    return logger, run_name

# 现在, 我们可以安全地从 'src' 导入模块了
try:
    from src.pipeline import TrainingPipeline
except ImportError:
    print("\n[错误] 无法导入 'src.pipeline.TrainingPipeline'。")
    print("\n--- 详细的导入错误跟踪 ---")  # <--- 添加这行
    traceback.print_exc()               # <--- 添加这行
    print("--------------------------\n")   # <--- 添加这行
    print("请确保您的项目结构如下:")
    print("GenRLMechSyn/")
    print("├── src/")
    print("│   ├── pipeline.py")
    print("├── scripts/")
    print("│   └── train.py")
    print("...")
    sys.exit(1)


def main():
    """
    主训练函数
    """
    # --- 启动日志记录 ---
    # (我们稍后会从 config 获取 run_name, 如果需要)
    logger, run_name_generated = setup_logging()

    logger.info(f"GenRLMechSyn 训练开始 - {run_name_generated}")
    
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="GenRLMechSyn - 训练主脚本")

    # 我们允许用户通过 --config 指定一个不同的配置文件
    # 默认值是 'configs/default_config.yaml' (相对于项目根目录)
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件的路径 (相对于项目根目录)'
    )

    args = parser.parse_args()

    # 2. 构建配置文件的绝对路径
    # (project_root / 'configs/default_config.yaml')
    config_path = os.path.join(project_root, args.config)

    # 3. 加载配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"--- 成功加载配置文件 ---")
        logger.info(f"路径: {config_path}")
        logger.info(f"项目名称: {config.get('project_name', 'N/A')}")
    except FileNotFoundError:
        logger.error(f"[致命错误] 配置文件未找到!")
        logger.error(f"尝试加载: {config_path}")
        logger.error("请确保 'configs/default_config.yaml' 文件存在于您的项目根目录中。")
        return
    except Exception as e:
        logger.error(f"[致命错误] 加载配置文件时出错: {e}")
        return

    # --- 固定随机数种子 ---
    # 推荐将种子值也放入 config 文件中管理
    seed_value = config.get('training', {}).get('random_seed', 42)  # 从配置获取, 默认 42
    set_seed(seed_value)

    # 4. 初始化并运行训练流程
    try:
        logger.info("--- 正在初始化训练流程 ---")
        pipeline = TrainingPipeline(config, project_root)

        logger.info("--- 训练流程开始 ---")
        pipeline.run()

    except KeyboardInterrupt:
        logger.warning("\n\n--- 训练被用户手动中断 ---")
        logger.warning("程序已停止。")
    except Exception as e:
        logger.error(f"--- [!!!] 训练流程中发生未捕获的致命错误 [!!!] ---")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"错误信息: {e}")
        logger.error("--- 详细堆栈跟踪 ---")
        traceback.print_exc()
        logger.error("--- 训练中止 ---")


if __name__ == "__main__":
    main()