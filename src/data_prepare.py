"""
data_prepare.py - KUAKE 数据加载与探查

注意：必须从项目根目录运行，或通过 run_pipeline.py 调用。
"""

import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.utils import *


def main():
    print("=" * 60)
    print("KUAKE 数据加载与探查")
    print("=" * 60)

    # 1. 加载 QTR 数据
    print("\n[1/4] 加载 KUAKE-QTR 数据...")
    qtr_train = load_qtr_data("train")
    qtr_dev = load_qtr_data("dev")
    qtr_test = load_qtr_data("test", has_label=False)
    print(f"  train: {len(qtr_train)} 条")
    print(f"  dev:   {len(qtr_dev)} 条")
    print(f"  test:  {len(qtr_test)} 条")

    # 2. 加载 IR 数据
    print("\n[2/4] 加载 KUAKE-IR 数据...")
    ir_corpus = load_ir_corpus()
    ir_queries = load_ir_queries("dev")
    ir_rel = load_ir_relevance("dev")
    print(f"  段落语料库: {len(ir_corpus)} 条")
    print(f"  dev queries: {len(ir_queries)} 条")
    print(f"  dev 相关性标注: {len(ir_rel)} 条")

    # 3. 生成统计报告
    print("\n[3/4] 生成统计报告...")
    stats = compute_dataset_stats(qtr_train, qtr_dev, ir_corpus, ir_rel)

    output_path = os.path.join(PROJ_ROOT, "output", "eval_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats)
    print(stats)
    print(f"\n报告已保存至: {output_path}")

    # 4. 保存预处理的 QTR 数据供后续使用
    print("\n[4/4] 保存预处理数据...")
    qtr_train.to_pickle(os.path.join(PROJ_ROOT, "models", "qtr_train.pkl"))
    qtr_dev.to_pickle(os.path.join(PROJ_ROOT, "models", "qtr_dev.pkl"))
    print("  数据已序列化保存")

    print("\n" + "=" * 60)
    print("数据探查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
