"""
infer_ir.py - KUAKE-IR 段落检索 + LR Rerank 全流程

流程：
  1. 加载 LR Rerank 模型
  2. 加载段落语料并构建 TF-IDF 向量化索引
  3. 加载 query
  4. TF-IDF 余弦相似度初筛：每个 query 召回 top-200 候选段落
  5. 用 LR Rerank 对候选段落重排序
  6. 输出 top-10 结果到 output/KUAKE-IR_dev_pred.tsv

注意：必须从项目根目录运行。
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from joblib import load, dump
from scipy.sparse import csr_matrix

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.utils import load_ir_corpus, load_ir_queries, load_ir_relevance


def build_tfidf_index(corpus_df, cache_dir):
    """构建并缓存 TF-IDF 向量化索引"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_path = os.path.join(cache_dir, "ir_tfidf_vectorizer.pkl")
    matrix_path = os.path.join(cache_dir, "ir_tfidf_matrix.npz")

    if os.path.exists(tfidf_path) and os.path.exists(matrix_path):
        print(f"  加载缓存的 TF-IDF 索引...")
        vectorizer = load(tfidf_path)
        from scipy.sparse import load_npz
        tfidf_matrix = load_npz(matrix_path)
        return vectorizer, tfidf_matrix

    print(f"  构建 TF-IDF 索引 (共 {len(corpus_df)} 篇文档)...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_df["passage"].values)
    print(f"    词汇表大小: {len(vectorizer.get_feature_names_out())}")
    print(f"    矩阵 shape: {tfidf_matrix.shape}")

    dump(vectorizer, tfidf_path)
    from scipy.sparse import save_npz
    save_npz(matrix_path, tfidf_matrix)
    return vectorizer, tfidf_matrix


def first_pass_retrieval(vectorizer, tfidf_matrix, query_text, top_k=200):
    """TF-IDF 余弦相似度检索"""
    query_vec = vectorizer.transform([query_text])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def main():
    MODEL_DIR = os.path.join(PROJ_ROOT, "models")
    OUTPUT_DIR = os.path.join(PROJ_ROOT, "output")
    CACHE_DIR = os.path.join(MODEL_DIR, "ir_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 参数
    TOP_K_BM25 = 200       # 第一阶段初筛保留数
    TOP_K_RERANK = 10      # Rerank 最终输出数
    N_SAMPLE_QUERIES = None  # 设为 None 跑全量 1000 query

    # 1. 加载 LR Rerank 模型
    print("[1/6] 加载 LR Rerank 模型...")
    model_path = os.path.join(MODEL_DIR, "lr_rerank.pkl")
    if not os.path.exists(model_path):
        print(f"  模型文件不存在: {model_path}")
        return
    rerank_pipe = load(model_path)
    print(f"  模型加载成功: {model_path}")

    # 2. 加载段落语料 + 构建TF-IDF索引
    print("\n[2/6] 加载段落语料库并构建 TF-IDF 索引...")
    corpus_df = load_ir_corpus()
    print(f"  段落语料: {len(corpus_df)} 条")
    vectorizer, tfidf_matrix = build_tfidf_index(corpus_df, CACHE_DIR)

    # 3. 加载 query
    print("\n[3/6] 加载 query...")
    queries = load_ir_queries("dev")
    print(f"  开发集 query 数: {len(queries)}")

    # 4. 加载标准答案
    print("\n[4/6] 加载标准答案...")
    relevance_df = load_ir_relevance("dev")
    relevance_dict = {}
    for _, row in relevance_df.iterrows():
        qid = row["query_id"]
        if qid not in relevance_dict:
            relevance_dict[qid] = set()
        relevance_dict[qid].add(int(row["doc_id"]))

    # 5. 检索 + Rerank
    print(f"\n[5/6] TF-IDF 初筛 (top-{TOP_K_BM25}) + LR Rerank...")

    selected_queries = list(queries.items())
    if N_SAMPLE_QUERIES is not None:
        selected_queries = selected_queries[:N_SAMPLE_QUERIES]
        print(f"  (演示模式: 仅处理前 {N_SAMPLE_QUERIES}/{len(queries)} 条 query)")

    all_results = []
    # 全量评估指标（基于初筛/重排的完整候选集合）
    first_pass_stats = {"ndcg10": [], "mrr": [], "recall10": [], "recall200": []}
    rerank_stats = {"ndcg10": [], "ndcg_all": [], "mrr": [], "recall10": []}
    import time

    for idx, (qid, query_text) in enumerate(selected_queries):
        t0 = time.time()

        # 第一阶段：TF-IDF 余弦相似度初筛 (top-200)
        top_indices, first_scores = first_pass_retrieval(
            vectorizer, tfidf_matrix, query_text, top_k=TOP_K_BM25
        )

        ground_truth = relevance_dict.get(qid, set())

        # ===== 初筛(top-200) 全量排序评估 =====
        first_docs = corpus_df.iloc[top_indices].copy()
        # 判定 200 个候选哪些命中标准答案
        first_relevant = np.array([1 if int(d) in ground_truth else 0
                                   for d in first_docs["doc_id"].values])
        first_recall10 = first_relevant[:10].sum() / len(ground_truth) if ground_truth else 0.0
        first_recall200 = first_relevant.sum() / len(ground_truth) if ground_truth else 0.0
        # 初筛 NDCG@10：若首个相关文档在 top-10 内则 NDCG=1，反之为 0
        first_ndcg10 = 1.0 if first_relevant[:10].sum() > 0 else 0.0
        first_mrr = 0.0
        for rank, rel in enumerate(first_relevant, 1):
            if rel == 1:
                first_mrr = 1.0 / rank
                break

        first_pass_stats["ndcg10"].append(first_ndcg10)
        first_pass_stats["mrr"].append(first_mrr)
        first_pass_stats["recall10"].append(first_recall10)
        first_pass_stats["recall200"].append(first_recall200)

        # ===== 构造 rerank 候选，LR 打分 =====
        candidates = first_docs.copy()
        candidates.rename(columns={"passage": "title"}, inplace=True)
        candidates["query"] = query_text
        candidates["label"] = -1  # dummy

        # 第二阶段：LR Rerank 对全部 top-200 打分并重排
        rerank_probs = rerank_pipe.predict_proba(candidates)
        candidates["score"] = rerank_probs[:, 2] + rerank_probs[:, 3]
        candidates = candidates.sort_values("score", ascending=False)

        # ===== Rerank 全量排序评估（top-200，不截断）=====
        rerank_relevant = np.array([1 if int(d) in ground_truth else 0
                                    for d in candidates["doc_id"].values])
        rerank_ndcg_all = 0.0
        if sum(rerank_relevant) > 0:
            from sklearn.metrics import ndcg_score
            k_all = len(candidates)
            rerank_ndcg_all = ndcg_score(
                rerank_relevant.reshape(1, -1), candidates["score"].values.reshape(1, -1), k=k_all)
        rerank_stats["ndcg_all"].append(rerank_ndcg_all)

        rerank_ndcg10 = 0.0
        if rerank_relevant[:10].sum() > 0:
            from sklearn.metrics import ndcg_score
            rerank_ndcg10 = ndcg_score(
                rerank_relevant[:10].reshape(1, -1), candidates["score"].values[:10].reshape(1, -1), k=10)
        rerank_stats["ndcg10"].append(rerank_ndcg10)

        rerank_mrr = 0.0
        for rank, rel in enumerate(rerank_relevant, 1):
            if rel == 1:
                rerank_mrr = 1.0 / rank
                break
        rerank_stats["mrr"].append(rerank_mrr)

        rerank_recall10 = rerank_relevant[:10].sum() / len(ground_truth) if ground_truth else 0.0
        rerank_stats["recall10"].append(rerank_recall10)

        # ===== 输出最终 top-10 =====
        top_k = candidates.head(TOP_K_RERANK)
        t1 = time.time()

        for _, row in top_k.iterrows():
            all_results.append({
                "query_id": qid,
                "doc_id": int(row["doc_id"]),
                "rerank_score": round(float(row["score"]), 6),
            })

        # 定期打日志
        n_total = len(selected_queries)
        if n_total >= 20 and (idx + 1) % max(1, n_total // 20) == 0:
            print(f"  已处理 {idx+1}/{n_total} query (当前耗时: {t1-t0:.2f}s/q)")
        elif n_total < 20:
            print(f"  处理 query {qid}: {t1-t0:.2f}s")

    # 6. 保存结果
    print("\n[6/6] 保存结果...")
    result_df = pd.DataFrame(all_results)

    tsv_path = os.path.join(OUTPUT_DIR, "KUAKE-IR_dev_pred.tsv")
    result_df[["query_id", "doc_id", "rerank_score"]].to_csv(
        tsv_path, sep="\t", header=False, index=False
    )
    print(f"  提交文件: {tsv_path} ({len(result_df)} 行)")

    # 评估指标（全量统计 + 初筛/重排对比）
    eval_summary = "\n" + "=" * 60
    eval_summary += "\nKUAKE-IR 检索 + LR Rerank 评估结果 (全量统计)"
    eval_summary += "\n" + "=" * 60
    eval_summary += f"\nTF-IDF 初筛 top-{TOP_K_BM25}, LR Rerank top-{TOP_K_RERANK}"
    eval_summary += f"\n评估 query 数: {len(selected_queries)}"

    # 初筛评估
    fp = {k: float(np.mean(v)) for k, v in first_pass_stats.items() if v}
    rk = {k: float(np.mean(v)) for k, v in rerank_stats.items() if v}

    eval_summary += "\n\n[初筛 - TF-IDF top-200 全量统计]"
    eval_summary += f"\n  NDCG@10: {fp.get('ndcg10', 0):.4f}"
    eval_summary += f"\n  MRR: {fp.get('mrr', 0):.4f}"
    eval_summary += f"\n  Recall@10: {fp.get('recall10', 0):.4f}"
    eval_summary += f"\n  Recall@200: {fp.get('recall200', 0):.4f}"

    eval_summary += "\n\n[Rerank - LR 重排 top-200 全量统计]"
    eval_summary += f"\n  NDCG@10: {rk.get('ndcg10', 0):.4f}"
    eval_summary += f"\n  NDCG@200 (全量不截断): {rk.get('ndcg_all', 0):.4f}"
    eval_summary += f"\n  MRR: {rk.get('mrr', 0):.4f}"
    eval_summary += f"\n  Recall@10: {rk.get('recall10', 0):.4f}"

    if fp.get('ndcg10') and fp['ndcg10'] > 0:
        eval_summary += \
            f"\n\n[Rerank 有效性] NDCG@10 初筛→重排: {fp['ndcg10']:.4f} → " \
            f"{rk.get('ndcg10', 0):.4f} (提升 {rk.get('ndcg10',0)-fp['ndcg10']:+.4f})"
    if fp.get('mrr') and fp['mrr'] > 0:
        eval_summary += \
            f"\n[Rerank 有效性] MRR 初筛→重排: {fp['mrr']:.4f} → " \
            f"{rk.get('mrr',0):.4f} (提升 {rk.get('mrr',0)-fp['mrr']:+.4f})"
    eval_summary += "\n" + "=" * 60
    print(eval_summary)

    # 追加到报告
    report_path = os.path.join(OUTPUT_DIR, "eval_report.txt")
    report_lines = []
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_lines = f.readlines()
    report_lines.append(eval_summary)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  评估报告: {report_path}")

    print("\n" + "=" * 60)
    print("KUAKE-IR 推理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
