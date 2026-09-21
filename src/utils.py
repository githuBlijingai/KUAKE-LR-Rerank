"""
KUAKE LR Rerank - 公共工具函数
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

# 基于文件位置动态计算项目根目录，避免硬编码本机绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_qtr_data(split: str = "train", has_label: bool = True) -> pd.DataFrame:
    """
    加载 KUAKE-QTR 数据
    split: "train" | "dev" | "test"
    has_label: 数据是否包含 label 字段（test 不包含）
    """
    path = os.path.join(PROJECT_ROOT, "KUAKE-QTR", f"KUAKE-QTR_{split}.json")
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
    path = os.path.join(PROJECT_ROOT, "KUAKE-IR", "corpus.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=["doc_id", "passage"])
    df["doc_id"] = df["doc_id"].astype(int)
    return df


def load_ir_queries(split: str = "train") -> Dict[int, str]:
    """
    加载 KUAKE-IR 的 query 列表
    split: "train" | "dev" | "test"
    """
    path = os.path.join(PROJECT_ROOT, "KUAKE-IR", f"KUAKE-IR_{split}_query.txt")
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
    path = os.path.join(PROJECT_ROOT, "KUAKE-IR", f"KUAKE-IR_{split}.tsv")
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
                            label_col: str = "label", k: Optional[int] = None,
                            relevant_threshold: int = 2) -> Dict[str, float]:
    """
    按 query 分组计算排序指标 (NDCG, MRR)，基于**全量**候选文档统计。

    df 需包含: query, score, label 三列。

    参数:
        k: NDCG 截断位置。None 表示不对文档数截断（使用每个 query 的全部文档，
           即全量 NDCG）；设置具体整数则对该 query 排序结果取前 k 个计算 NDCG@k。
        relevant_threshold: 判定"相关"的 label 阈值，>= 该值的文档计为相关（用于 MRR）。

    说明:
        对每个 query，先将该 query 的**全部**候选文档按 score 降序排列，再计算排序指标，
        而非只取 head(k)。这样 NDCG 能反映模型对完整候选集合的排序能力。
    """
    from sklearn.metrics import ndcg_score
    ndcg_list = []
    mrr_list = []

    for qid, group in df.groupby("query"):
        # 全量排序（不再在这里截断文档数）
        group = group.sort_values(score_col, ascending=False)
        y_true_all = group[label_col].values
        y_score_all = group[score_col].values

        # 对排序结果取前 k 个作为 NDCG@k 的评估窗口；k=None 用全量
        if k is not None:
            y_true = y_true_all[:k]
            y_score = y_score_all[:k]
        else:
            y_true = y_true_all
            y_score = y_score_all

        # 至少要有 2 个文档才计算 NDCG（sklearn 要求）
        if len(y_true) >= 2:
            ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
            ndcg_list.append(ndcg)

        # MRR: 第一个相关文档 (label >= relevant_threshold) 在**全量**排序中的位置倒数
        for rank, label in enumerate(y_true_all, 1):
            if label >= relevant_threshold:
                mrr_list.append(1.0 / rank)
                break
        else:
            mrr_list.append(0.0)

    metrics = {"MRR": float(np.mean(mrr_list))}
    if ndcg_list:
        metric_name = "NDCG" if k is None else f"NDCG@{k}"
        metrics[metric_name] = float(np.mean(ndcg_list))
    return metrics
