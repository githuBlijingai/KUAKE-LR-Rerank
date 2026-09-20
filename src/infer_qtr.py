"""
infer_qtr.py - KUAKE-QTR 测试集推理

用训练好的 LR Rerank 模型对 KUAKE-QTR 测试集进行预测，
生成 CBLUE 标准提交格式（保留原 id、query、title 并添加 predict_label 字段）。

注意：必须从项目根目录运行。
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from joblib import load

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.utils import load_qtr_data


def main():
    MODEL_DIR = os.path.join(PROJ_ROOT, "models")
    OUTPUT_DIR = os.path.join(PROJ_ROOT, "output")

    # 1. 加载模型
    print("[1/4] 加载模型...")
    model_path = os.path.join(MODEL_DIR, "lr_rerank.pkl")
    if not os.path.exists(model_path):
        print(f"  模型文件不存在: {model_path}，请先运行 train_eval.py")
        return
    best_pipe = load(model_path)
    print(f"  模型加载成功: {model_path}")

    # 2. 加载测试集
    print("\n[2/4] 加载测试集...")
    test_df = load_qtr_data("test", has_label=False)
    print(f"  测试集: {len(test_df)} 条")

    # 3. 预测
    print("\n[3/4] 进行预测...")
    preds = best_pipe.predict(test_df)
    probs = best_pipe.predict_proba(test_df)

    test_df = test_df.copy()
    test_df["predict_label"] = preds.astype(int)
    test_df["predict_score"] = probs[:, 2] + probs[:, 3]

    label_dist = test_df["predict_label"].value_counts().sort_index()
    print(f"  预测分布: {dict(label_dist)}")

    # 4. 保存提交文件
    print("\n[4/4] 保存提交文件...")
    out_path = os.path.join(OUTPUT_DIR, "KUAKE-QTR_test_pred.json")

    records = []
    for _, row in test_df.iterrows():
        records.append({
            "id": row["id"],
            "query": row["query"],
            "title": row["title"],
            "predict_label": int(row["predict_label"]),
            "predict_score": round(float(row["predict_score"]), 4),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  提交文件已保存: {out_path} ({len(records)} 条)")

    # 统计
    print("\n" + "=" * 60)
    print("KUAKE-QTR 推理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
