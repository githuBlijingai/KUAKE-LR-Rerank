"""
KUAKE LR Rerank - 公共工具函数
"""

import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = r"C:\Users\Administrator.DESKTOP-VIVLMOS\Desktop\Chinese_Medical_QA_Dataset-fe7526dbba0fa3264e28792d85662e6415a1d409"


def load_qtr_data(split: str = "train", has_label: bool = True) -> pd.DataFrame:
    """
    加载 KUAKE-QTR 数据
    split: "train" | "dev" | "test"
    has_label: 数据是否包含 label 字段（test 不包含）
    """
    path = f"{PROJECT_ROOT}\\KUAKE-QTR\\KUAKE-QTR_{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if has_label:
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = -1  # 占位符
    return df


def load_ir_corpus() -> pd.DataFrame:
    """加载 KUAKE-IR 的段落语料库 (corpus.tsv)"""
    path = f"{PROJECT_ROOT}\\KUAKE-IR\\corpus.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=["doc_id", "passage"])
    df["doc_id"] = df["doc_id"].astype(int)
    return df


def load_ir_queries(split: str = "train") -> Dict[int, str]:
    """
    加载 KUAKE-IR 的 query 列表
    split: "train" | "dev" | "test"
    """
    path = f"{PROJECT_ROOT}\\KUAKE-IR\\KUAKE-IR_{split}_query.txt"
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                qid, query = parts
                queries[int(qid)] = query
    return queries


def load_ir_relevance(split: str = "dev") -> pd.DataFrame:
    """
    加载 KUAKE-IR 的 query-doc 相关性标注
    split: "dev" | "train"
    """
    path = f"{PROJECT_ROOT}\\KUAKE-IR\\KUAKE-IR_{split}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=["query_id", "doc_id"])
    df["query_id"] = df["query_id"].astype(int)
    df["doc_id"] = df["doc_id"].astype(int)
    return df


def compute_dataset_stats(qtr_train: pd.DataFrame, qtr_dev: pd.DataFrame,
                          ir_corpus: pd.DataFrame, ir_rel: pd.DataFrame) -> str:
    """计算数据集统计信息，返回报告文本"""
    lines = []
    lines.append("=" * 60)
    lines.append("KUAKE 数据集统计信息")
    lines.append("=" * 60)

    lines.append(f"\n--- KUAKE-QTR ---")
    lines.append(f"训练集: {len(qtr_train)} 条")
    lines.append(f"验证集: {len(qtr_dev)} 条")

    # Label 分布
    for name, df in [("训练集", qtr_train), ("验证集", qtr_dev)]:
        lines.append(f"\n{name} label 分布:")
        label_counts = df["label"].value_counts().sort_index()
        for lbl in [0, 1, 2, 3]:
            cnt = label_counts.get(lbl, 0)
            pct = cnt / len(df) * 100
            lines.append(f"  label={lbl}: {cnt} ({pct:.1f}%)")

    # 长度统计
    qtr_train["query_len"] = qtr_train["query"].str.len()
    qtr_train["title_len"] = qtr_train["title"].str.len()
    lines.append(f"\nQuery 长度 (字符): mean={qtr_train['query_len'].mean():.1f}, "
                 f"median={qtr_train['query_len'].median():.0f}, "
                 f"max={qtr_train['query_len'].max()}")
    lines.append(f"Title 长度 (字符): mean={qtr_train['title_len'].mean():.1f}, "
                 f"median={qtr_train['title_len'].median():.0f}, "
                 f"max={qtr_train['title_len'].max()}")

    # 唯一 query 数
    unique_queries = qtr_train["query"].nunique()
    lines.append(f"\n唯一 query 数: {unique_queries}")
    lines.append(f"平均每个 query 对应的 title 数: {len(qtr_train) / unique_queries:.2f}")

    lines.append(f"\n--- KUAKE-IR ---")
    lines.append(f"候选段落库: {len(ir_corpus)} 条")
    lines.append(f"段落平均长度: {ir_corpus['passage'].str.len().mean():.1f} 字符")

    lines.append(f"\nKUAKE-IR 验证集 query-doc 相关标注: {len(ir_rel)} 条")
    unique_q = ir_rel["query_id"].nunique()
    lines.append(f"唯一 query 数: {unique_q}")
    lines.append(f"每个 query 平均相关段落数: {len(ir_rel) / unique_q:.1f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """计算 NDCG@K"""
    from sklearn.metrics import ndcg_score
    return ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)


def compute_ranking_metrics(df: pd.DataFrame, score_col: str = "score",
                            label_col: str = "label", k: int = 10) -> Dict[str, float]:
    """
    按 query 分组计算排序指标 (NDCG, MRR)
    df 需包含: query, score, label 三列
    """
    from sklearn.metrics import ndcg_score
    ndcg_list = []
    mrr_list = []

    for qid, group in df.groupby("query"):
        group = group.sort_values(score_col, ascending=False).head(k)
        y_true = group[label_col].values.reshape(1, -1)
        y_score = group[score_col].values.reshape(1, -1)

        if len(y_true.flatten()) >= 2:
            ndcg = ndcg_score(y_true, y_score, k=min(k, len(y_true.flatten())))
            ndcg_list.append(ndcg)

        # MRR: 第一个相关文档 (label>=2) 的位置倒数
        for rank, label in enumerate(group[label_col].values, 1):
            if label >= 2:
                mrr_list.append(1.0 / rank)
                break
        else:
            mrr_list.append(0.0)

    return {
        f"NDCG@{k}": float(np.mean(ndcg_list)),
        "MRR": float(np.mean(mrr_list)),
    }
