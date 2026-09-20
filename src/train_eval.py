"""
train_eval.py - 训练 + 超参数调优 + 评估

训练流程：
  1. 加载 QTR 训练集/验证集
  2. 超参数 GridSearch
  3. 最终模型训练
  4. 验证集评估（Accuracy, NDCG, MRR）
  5. 保存模型 & 特征重要性图

注意：必须从项目根目录运行，或通过 run_pipeline.py 调用。
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.utils import load_qtr_data, compute_ranking_metrics
from src.feature_engineering import build_full_pipeline


def main():
    OUTPUT_DIR = os.path.join(PROJ_ROOT, "output")
    MODEL_DIR = os.path.join(PROJ_ROOT, "models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("KUAKE-QTR LR Rerank 训练与评估")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_df = load_qtr_data("train")
    dev_df = load_qtr_data("dev")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(dev_df)} 条")

    # 2. 超参数搜索
    print("\n[2/6] 超参数搜索 (GridSearchCV, 3折)...")
    param_grid = {
        "features__query_tfidf__tfidf__max_features": [2000, 4000],
        "features__title_tfidf__tfidf__max_features": [2000, 4000],
        "features__concat_tfidf__tfidf__max_features": [2000, 4000],
        "features__manual__scaler__with_mean": [True],
        "clf__C": [0.5, 1.0],
    }

    base_pipe = build_full_pipeline()
    group_kfold = GroupKFold(n_splits=3)
    groups = train_df["query"]

    grid = GridSearchCV(
        base_pipe,
        param_grid,
        cv=group_kfold.split(train_df, train_df["label"], groups),
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )

    grid.fit(train_df, train_df["label"])
    print(f"\n最佳参数: {grid.best_params_}")
    print(f"最佳 CV Accuracy: {grid.best_score_:.4f}")

    best_pipe = grid.best_estimator_

    # 3. 验证集评估
    print("\n[3/6] 验证集评估...")
    dev_preds = best_pipe.predict(dev_df)
    dev_probs = best_pipe.predict_proba(dev_df)

    acc = accuracy_score(dev_df["label"], dev_preds)
    print(f"  Accuracy: {acc:.4f}")

    print(f"\n  分类报告:")
    print(classification_report(dev_df["label"], dev_preds,
                                target_names=["不相关(0)", "弱相关(1)", "中等相关(2)", "强相关(3)"]))

    # 混淆矩阵
    cm = confusion_matrix(dev_df["label"], dev_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0-不相关", "1-弱相关", "2-中等", "3-强相关"],
                yticklabels=["0-不相关", "1-弱相关", "2-中等", "3-强相关"])
    plt.title("KUAKE-QTR LR Rerank 混淆矩阵 (验证集)")
    plt.xlabel("预测")
    plt.ylabel("真实")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵已保存: {cm_path}")

    # 排序指标
    print("\n  排序指标 (按 query 分组评估):")
    dev_df_eval = dev_df.copy()
    dev_df_eval["score"] = dev_probs[:, 2] + dev_probs[:, 3]
    ranking_metrics = compute_ranking_metrics(dev_df_eval, k=10)
    for metric, value in ranking_metrics.items():
        print(f"    {metric}: {value:.4f}")

    # 4. 特征重要性
    print("\n[4/6] 特征重要性分析...")
    lr = best_pipe.named_steps["clf"]

    feature_union = best_pipe.named_steps["features"]
    feature_names = []
    for name, transformer, _ in feature_union.transformers:
        if name == "manual":
            manual_names = [
                "词重叠率", "Jaccard系数", "Query命中Title词比",
                "Title命中Query词比", "编辑距离(归一化)", "最长公共子串长度",
                "最长公共子串比率", "数字匹配数", "长度比", "长度差", "Query长度",
                "BM25分数"
            ]
            feature_names.extend([f"manual_{n}" for n in manual_names])
        else:
            tfidf = transformer.named_steps.get("tfidf", None)
            if tfidf is not None and hasattr(tfidf, "get_feature_names_out"):
                try:
                    names = tfidf.get_feature_names_out()
                    feature_names.extend([f"{name}_{n}" for n in names])
                except:
                    pass

    if lr.coef_.ndim == 2 and lr.coef_.shape[0] > 1:
        coef_importance = np.mean(np.abs(lr.coef_), axis=0)
    else:
        coef_importance = np.abs(lr.coef_).flatten()

    n_features_actual = len(coef_importance)
    n_names = len(feature_names)

    if n_names == n_features_actual:
        fi_df = pd.DataFrame({"feature": feature_names, "importance": coef_importance})
    else:
        print(f"  特征名称数量 ({n_names}) 与系数数量 ({n_features_actual}) 不匹配，使用序号")
        fi_df = pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n_features_actual)],
            "importance": coef_importance
        })

    fi_df = fi_df.sort_values("importance", ascending=False)
    top_n = min(30, len(fi_df))
    top_features = fi_df.head(top_n)

    plt.figure(figsize=(10, 10))
    plt.barh(range(top_n), top_features["importance"].values[::-1])
    plt.yticks(range(top_n), top_features["feature"].values[::-1])
    plt.xlabel("平均 |系数| (多分类平均)")
    plt.title(f"KUAKE-QTR LR Rerank 特征重要性 Top-{top_n}")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  特征重要性图已保存: {fi_path}")

    print(f"\n  Top-10 特征:")
    for i, row in fi_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.6f}")

    # 5. 保存模型
    print("\n[5/6] 保存模型...")
    model_path = os.path.join(MODEL_DIR, "lr_rerank.pkl")
    dump(best_pipe, model_path)
    print(f"  模型已保存: {model_path}")

    # 6. 更新评估报告
    print("\n[6/6] 更新评估报告...")
    report_path = os.path.join(OUTPUT_DIR, "eval_report.txt")
    report_lines = []
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_lines = f.readlines()

    report_lines.append("\n" + "=" * 60)
    report_lines.append("LR Rerank 训练评估结果")
    report_lines.append("=" * 60)
    report_lines.append(f"\n最佳超参数: {grid.best_params_}")
    report_lines.append(f"最佳 CV Accuracy: {grid.best_score_:.4f}")
    report_lines.append(f"\n验证集 Accuracy: {acc:.4f}")
    for metric, value in ranking_metrics.items():
        report_lines.append(f"验证集 {metric}: {value:.4f}")
    report_lines.append(f"\n验证集分类报告:")
    report_lines.append(classification_report(dev_df["label"], dev_preds))
    report_lines.append(f"\nTop-10 特征:")
    for i, row in fi_df.head(10).iterrows():
        report_lines.append(f"  {row['feature']}: {row['importance']:.6f}")
    report_lines.append("\n" + "=" * 60)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  评估报告已更新: {report_path}")

    print("\n" + "=" * 60)
    print("训练与评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
