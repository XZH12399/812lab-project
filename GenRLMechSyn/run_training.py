# run_training.py (放在项目根目录下)

import os
import sys
import subprocess
from pathlib import Path


def main():
    # 1. 设置 PYTHONHASHSEED
    seed_value = '0'  # 通常设为 '0' 来禁用哈希随机化
    print(f"--- 设置 PYTHONHASHSEED={seed_value} ---")

    # 2. 获取当前环境并更新
    current_env = os.environ.copy()
    current_env['PYTHONHASHSEED'] = seed_value

    # 3. 确定要运行的脚本路径
    project_root = Path(__file__).parent.resolve()
    train_script_path = project_root / "scripts" / "train.py"

    if not train_script_path.exists():
        print(f"[错误] 找不到训练脚本: {train_script_path}")
        sys.exit(1)

    # 4. 确定 Python 解释器路径
    #    使用 sys.executable 确保使用与运行此脚本相同的解释器
    python_executable = sys.executable

    # 5. 构建命令行参数 (与您在终端运行时相同)
    #    例如，如果您需要传递 --config 参数，可以在这里添加
    command = [
        python_executable,
        str(train_script_path),
        # '--config', 'configs/another_config.yaml' # 如果需要传递参数
    ]

    print(f"--- 即将运行: {' '.join(command)} ---")

    # 6. 使用 subprocess 运行脚本，并传入修改后的环境
    try:
        # --- 核心修改: 不再捕获输出, 让子进程直接打印到终端 ---
        result = subprocess.run(
            command,
            env=current_env,
            # capture_output=False, # 可以省略, 默认不捕获
            # text=False,          # 省略
            check=False
        )

        # 7. --- 核心修改: 不再打印捕获的输出 ---
        # print("\n--- 子进程 (train.py) 标准输出 ---")
        # print(result.stdout) # 移除或注释掉

        if result.returncode != 0:
            # print("\n--- 子进程 (train.py) 标准错误 ---")
            # print(result.stderr) # 移除或注释掉
            # stderr 的内容现在会直接显示在终端
            print(f"\n[错误] train.py 运行失败，返回代码: {result.returncode}")
        else:
            print("\n--- train.py 运行成功 ---")

    except Exception as e:
        print(f"\n[致命错误] 运行子进程时出错: {e}")


if __name__ == "__main__":
    main()
