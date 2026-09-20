"""
02_feature_engineering.py - 特征工程与 Pipeline 定义

特征体系：
  1. 手工特征：词重叠、Jaccard、编辑距离、最长公共子串、数字匹配、长度特征
  2. BM25 分数
  3. TF-IDF: query char_wb, title char_wb, concat char
"""

import numpy as np
import pandas as pd
import jieba
import re
from typing import List, Tuple, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─── 手工特征提取器 ───

def _tokenize(text: str) -> List[str]:
    """分词（带空返回空列表）"""
    if not isinstance(text, str) or not text.strip():
        return []
    return list(jieba.cut(text))


def _extract_digits(text: str) -> set:
    """提取文本中的数字"""
    return set(re.findall(r'\d+\.?\d*', str(text)))


def _bm25_score_single(query_tokens: List[str], title_tokens: List[str],
                        k1: float = 1.5, b: float = 0.75) -> float:
    """计算单个 query vs title 的 BM25 分数（不依赖全局文档集）"""
    if not query_tokens or not title_tokens:
        return 0.0
    from collections import Counter
    query_counts = Counter(query_tokens)
    title_counts = Counter(title_tokens)
    n = len(title_tokens)

    avg_dl = n  # 单文档时 avg_dl = 文档长度
    score = 0.0
    for term, qf in query_counts.items():
        tf = title_counts.get(term, 0)
        if tf == 0:
            continue
        idf = np.log(1 + (n - tf + 0.5) / (tf + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * n / max(avg_dl, 1)))
    return score


def compute_manual_features(X: np.ndarray) -> np.ndarray:
    """
    计算手工特征 + BM25 分数
    X: shape (n_samples, 2) 其中每行是 [query, title]
    返回: shape (n_samples, 12) 的特征矩阵 (11 手工 + 1 BM25)
    """
    features = []
    for query, title in X:
        query = str(query) if query else ""
        title = str(title) if title else ""

        q_words = set(_tokenize(query))
        t_words = set(_tokenize(title))
        q_tokens = _tokenize(query)
        t_tokens = _tokenize(title)

        q_len = len(query)
        t_len = len(title)

        # 1. 词重叠率: |Q∩T| / min(|Q|,|T|)
        overlap = len(q_words & t_words)
        overlap_ratio = overlap / max(min(len(q_words), len(t_words)), 1)

        # 2. Jaccard 系数
        union = len(q_words | t_words)
        jaccard = overlap / max(union, 1)

        # 3. Query 命中 Title 的词比例
        q_hit_t = sum(1 for w in q_words if w in title) / max(len(q_words), 1)

        # 4. Title 命中 Query 的词比例
        t_hit_q = sum(1 for w in t_words if w in query) / max(len(t_words), 1)

        # 5. 编辑距离归一化
        from Levenshtein import distance as lev_dist
        edit_dist = lev_dist(query, title)
        edit_norm = 1.0 - edit_dist / max(q_len, t_len, 1)

        # 6. 最长公共子串长度
        def longest_common_substring(s1: str, s2: str) -> int:
            if not s1 or not s2:
                return 0
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            max_len = 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        max_len = max(max_len, dp[i][j])
            return max_len

        lcs_len = longest_common_substring(query, title)

        # 7. 最长公共子串比率
        lcs_ratio = lcs_len / max(q_len, t_len, 1)

        # 8. 数字匹配数
        q_digits = _extract_digits(query)
        t_digits = _extract_digits(title)
        digit_match_count = len(q_digits & t_digits)

        # 9. 长度比
        length_ratio = q_len / max(t_len, 1)

        # 10. 长度差
        length_diff = abs(q_len - t_len)

        # 11. Query 长度
        query_len = q_len

        # 12. BM25 分数 (单文档 BM25, 无全局依赖)
        bm25_sc = _bm25_score_single(q_tokens, t_tokens)

        features.append([
            overlap_ratio, jaccard, q_hit_t, t_hit_q,
            edit_norm, lcs_len, lcs_ratio,
            digit_match_count, length_ratio, length_diff, query_len,
            bm25_sc
        ])

    return np.array(features, dtype=np.float64)


# ─── BM25 已合并到 compute_manual_features 中 ───


# ─── 列提取器 ───

def extract_query(df: pd.DataFrame) -> pd.Series:
    return df["query"]

def extract_title(df: pd.DataFrame) -> pd.Series:
    return df["title"]

def extract_concat(df: pd.DataFrame) -> pd.Series:
    return df["query"] + " " + df["title"]

def extract_query_title_pairs(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["query"].values, df["title"].values])


# ─── 构建完整 Pipeline ───

def build_feature_union(max_features: int = 5000) -> FeatureUnion:
    """构建特征联合"""

    manual_feats = FunctionTransformer(
        extract_query_title_pairs,
        validate=False
    )

    return FeatureUnion([
        # 1. 手工特征 + BM25 (词重叠, Jaccard, 编辑距离, 公共子串, 数字匹配, 长度, BM25)
        ("manual", Pipeline([
            ("extract", manual_feats),
            ("compute", FunctionTransformer(compute_manual_features, validate=False)),
            ("scaler", StandardScaler()),
        ])),

        # 2. Query TF-IDF (char_wb, ngram 1-3)
        ("query_tfidf", Pipeline([
            ("extract", FunctionTransformer(extract_query, validate=False)),
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(1, 3),
                max_features=max_features,
                sublinear_tf=True,
            )),
        ])),

        # 3. Title TF-IDF (char_wb, ngram 1-3)
        ("title_tfidf", Pipeline([
            ("extract", FunctionTransformer(extract_title, validate=False)),
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(1, 3),
                max_features=max_features,
                sublinear_tf=True,
            )),
        ])),

        # 4. Concat TF-IDF (char, ngram 1-2)
        ("concat_tfidf", Pipeline([
            ("extract", FunctionTransformer(extract_concat, validate=False)),
            ("tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 2),
                max_features=max_features,
                sublinear_tf=True,
            )),
        ])),
    ])


def build_full_pipeline(max_features: int = 5000, C: float = 1.0,
                        class_weight: str = "balanced", max_iter: int = 1000) -> Pipeline:
    """构建完整 Pipeline: FeatureUnion → LR"""
    return Pipeline([
        ("features", build_feature_union(max_features=max_features)),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])


if __name__ == "__main__":
    # 快速测试：加载少量数据验证特征提取
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_qtr_data

    print("测试特征提取（使用训练集前 100 条）...")
    df = load_qtr_data("train").head(100)

    X_pairs = extract_query_title_pairs(df)
    manual_feats = compute_manual_features(X_pairs)
    print(f"手工特征矩阵 shape: {manual_feats.shape}")
    print(f"特征列名: overlap_ratio, jaccard, q_hit_t, t_hit_q, edit_norm, "
          f"lcs_len, lcs_ratio, digit_match, len_ratio, len_diff, query_len")

    # 测试 BM25
    print("\n测试 BM25 特征...")
    bm25_ext = BM25FeatureExtractor()
    bm25_ext.fit(X_pairs)
    bm25_scores = bm25_ext.transform(X_pairs)
    print(f"BM25 分数 shape: {bm25_scores.shape}")
    print(f"BM25 分数范围: {bm25_scores.min():.4f} ~ {bm25_scores.max():.4f}")

    # 测试完整 Pipeline
    print("\n测试完整 Pipeline 训练...")
    pipe = build_full_pipeline(max_features=1000)
    pipe.fit(df, df["label"])
    preds = pipe.predict(df.head(10))
    probs = pipe.predict_proba(df.head(10))
    print(f"预测结果: {preds}")
    print(f"预测概率 shape: {probs.shape}")
    print("特征工程 Pipeline 测试通过！")
