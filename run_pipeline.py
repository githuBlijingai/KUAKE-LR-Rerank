"""
启动脚本 - 从项目根目录运行各个处理步骤
使用方式：python run_pipeline.py --step train
"""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJ_ROOT, "src")

# 确保根目录和 src 都在 path 中（src 用于包导入）
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# 使用 subprocess 运行而不是直接 import，避免复杂路径问题
import subprocess


def run_module(module_name: str):
    """运行 src 目录下的模块"""
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, {repr(PROJ_ROOT)})
os.chdir({repr(PROJ_ROOT)})
from src.{module_name} import main
main()
        """],
        capture_output=False,
        cwd=PROJ_ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KUAKE LR Rerank Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["data_prepare", "train", "infer_qtr", "infer_ir", "all"])
    args = parser.parse_args()

    if args.step in ("data_prepare", "all"):
        print("\n" + "=" * 60)
        print("步骤 1: 数据加载与探查")
        print("=" * 60)
        run_module("data_prepare")

    if args.step in ("train", "all"):
        print("\n" + "=" * 60)
        print("步骤 2: 训练与评估")
        print("=" * 60)
        run_module("train_eval")

    if args.step in ("infer_qtr", "all"):
        print("\n" + "=" * 60)
        print("步骤 3: KUAKE-QTR 测试集推理")
        print("=" * 60)
        run_module("infer_qtr")

    if args.step in ("infer_ir", "all"):
        print("\n" + "=" * 60)
        print("步骤 4: KUAKE-IR 检索 + LR Rerank")
        print("=" * 60)
        run_module("infer_ir")

    print("\n✓ 所有步骤完成！")
