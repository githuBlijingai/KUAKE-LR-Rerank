# KUAKE 医学段落检索 LR Rerank

基于 sklearn LogisticRegression 实现的轻量化 **Pointwise Rerank**（重排序模型），在 CBLUE 基准的 KUAKE-QTR（查询-标题相关性）和 KUAKE-IR（医学段落检索）数据集上完成训练、评估与推理。

> 📖 详细的**全流程图文技术报告**请参见 [`lr_rerank_report.html`](./lr_rerank_report.html)，包含 Rerank 概念图解、Pipeline 架构图、12 维特征详解、两阶段检索流程图等面向小白的通俗讲解。

---

## 什么是 Rerank？

**一句话理解**：Rerank 就是对搜索引擎初步召回的结果进行二次排序，把最相关的排到最前面。

> 想象去图书馆找书——第一步是**检索**：管理员从书架中快速挑出可能相关的 200 本书。第二步是 **Rerank**：你翻看标题和摘要，把最相关的 10 本挑出来放到桌上。

本项目使用 **Pointwise 方法**：对每个 query-document 对独立打分，按分数从高到低排列。Rerank 方法对比：

| 方法 | 思想 | 复杂度 | 代表 |
|------|------|--------|------|
| **Pointwise** | 对每个 query-doc 对独立打分 | 低 | Logistic Regression、BERT 单塔 |
| Pairwise | 比较文档对的相对顺序 | 中 | RankNet、LambdaRank |
| Listwise | 直接优化整个排序列表的指标 | 高 | ListNet、LambdaMART |

## 项目结构

```
├── KUAKE-QTR/                      # QTR 数据集（需自行从 CBLUE 下载，不入库）
│   ├── KUAKE-QTR_train.json        # 训练集 (24,174 条)
│   ├── KUAKE-QTR_dev.json          # 验证集 (2,913 条)
│   └── KUAKE-QTR_test.json         # 测试集 (5,465 条)
├── KUAKE-IR/                       # IR 数据集（需自行从 CBLUE 下载，不入库）
│   ├── corpus.tsv                  # 段落库 (958,846 篇)
│   ├── KUAKE-IR_train.tsv          # IR 训练标注
│   ├── KUAKE-IR_dev.tsv            # IR 验证标注 (1,000 条)
│   └── KUAKE-IR_dev_query.txt      # 验证集查询 (1,000 条)
├── src/
│   ├── utils.py                    # 数据加载、评估指标工具函数
│   ├── data_prepare.py             # 数据统计与探查
│   ├── feature_engineering.py      # 特征工程 & Pipeline 定义
│   ├── train_eval.py               # 训练 + GridSearch + 评估
│   ├── infer_qtr.py                # KUAKE-QTR 测试集推理
│   └── infer_ir.py                 # KUAKE-IR 检索 + Rerank
├── run_pipeline.py                 # 一键运行脚本（python run_pipeline.py --step train）
├── models/
│   ├── lr_rerank.pkl               # 训练好的 LR Pipeline（1.7 GB，LFS 跟踪）
│   └── ir_cache/                   # IR TF-IDF 向量化缓存
│       ├── ir_tfidf_vectorizer.pkl
│       └── ir_tfidf_matrix.npz
├── output/
│   ├── eval_report.txt             # 完整评估报告
│   ├── confusion_matrix.png        # 混淆矩阵（验证集）
│   ├── feature_importance.png      # 特征重要性 Top-30
│   ├── KUAKE-QTR_test_pred.json    # QTR 测试集预测 (5,465 条)
│   ├── KUAKE-IR_dev_pred.tsv       # IR 检索排序结果 (10,000 行)
│   └── lr_rerank_report.html       # 全流程技术报告（含图文 SVG 图解）
└── README.md                       # 本文档
```

> 数据说明：`KUAKE-QTR/`、`KUAKE-IR/` 目录及 zip 压缩包体积大，未纳入版本库（已在 `.gitignore` 排除），请从 [CBLUE 数据集](https://github.com/aliyun/research-duty) 下载后放入项目根目录。所有路径均基于 `__file__` 动态计算，克隆后无需修改任何路径即可直接运行。

---

## 数据集

| 数据集 | 用途 | 规模 | 标签 |
|--------|------|------|------|
| **KUAKE-QTR** | Query-Title 相关性分类 | 24,174 train / 2,913 dev / 5,465 test | 4 分类 (0=不相关, 1=略相关, 2=较相关, 3=完全相关) |
| **KUAKE-IR** | 段落检索 | 958,846 篇段落库 / 1,000 查询标注 | query-doc 相关/不相关 |

QTR 训练集 label 分布：label=0 (16.1%) / label=1 (22.3%) / label=2 (22.8%) / label=3 (38.9%)，其中完全相关占比最高。

---

## 核心概念（面向小白）

### TF-IDF（词频-逆文档频率）
衡量一个词在文档中的重要性。TF = 词在文档中出现的次数，IDF = 词在整个语料中的稀缺程度。本项目使用 `char_wb` 分析器在字符级别上做 ngram(1~3)，设置 `sublinear_tf=True`（对数平滑）。

### BM25（Best Matching 25）
信息检索中最常用的排序函数，是 TF-IDF 的改进版。引入了两个参数：**k1** 控制词频饱和度，**b** 控制文档长度归一化。本项目实现了一个轻量版 BM25，基于 Counter 统计词频对单个文档对计算分数。

### Logistic Regression（逻辑回归）
一种广义线性模型，通过 Softmax 函数将线性输出映射为概率分布，适用于多分类问题。相比深度模型，LR 训练快、可解释性强（特征权重直接反映重要性）。

### NDCG（归一化折损累计增益）
排序质量的核心指标，不仅考虑"相关文档是否被召回"，还考虑排序位置——越靠前权重越高。NDCG@10 = 1.0 表示完美排序。

本项目的 NDCG 评估采用 **全量统计**：
- 对每个 query，将**全部**候选文档按预测分数从高到低排序（而非只取前 k 个）
- NDCG@10 = 对全量排序结果截断到前 10 个位置计算
- 全量 NDCG（k=None）= 对不截断的完整候选列表计算，反映模型对完整候选集合的排序能力
- IR 场景还会**对比初筛与 rerank 的 NDCG 提升量**，直观验证重排有效性

### MRR（平均倒数排名）
第一个相关文档所在位置倒数的平均值。第一位 = 1.0，第二位 = 0.5，反映模型将最相关内容排到最前面的能力。

---

## 方法

### Pipeline 架构

通过 `sklearn FeatureUnion` **并行计算 4 组特征**，拼接后送入 `LogisticRegression`（multinomial softmax），取 P(label≥2) 作为 Rerank 排序分数：

```
              ┌──────────────┐
              │ query + title│
              └──────┬───────┘
         ┌───────────┼───────────────────┐
         ▼           ▼           ▼       ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ 手动特征 │ │Query    │ │Title    │ │Concat    │
   │ (12维)  │ │TF-IDF   │ │TF-IDF   │ │TF-IDF    │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘
        └───────────┼───────────┼───────────┘
                    ▼
           ┌────────────────┐
           │  FeatureUnion  │
           │ (拼接～15000维) │
           └───────┬────────┘
                   ▼
          ┌────────────────┐
          │ LogisticRegr.  │
          │ Softmax 4分类  │
          └───────┬────────┘
                   ▼
          ┌────────────────┐
          │ Rerank 排序分数 │
          │ P(label≥2)     │
          └────────────────┘
```

#### 12 维手动特征详解

| # | 特征 | 说明 | 含义 |
|---|------|------|------|
| 1 | overlap_ratio | query 和 title 公共 token 数 / query token 数 | 词重叠比例 |
| 2 | jaccard | 公共 token 数 / 并集 token 数 | 集合相似度 |
| 3 | q_hit_title | query 命中 title 的 token 比例 | Query 对 Title 的覆盖率 |
| 4 | t_hit_query | title 命中 query 的 token 比例 | Title 对 Query 的覆盖率 |
| 5 | edit_dist_norm | 归一化编辑距离 | 字符串相似度（Levenshtein） |
| 6 | lcs_len | 最长公共子串长度 | 连续相同字符数 |
| 7 | lcs_ratio | 最长公共子串 / min(len1, len2) | 连续相同比例 |
| 8 | digit_match | 相同数字个数 | 数字一致性（如剂量、时间） |
| 9 | len_ratio | title_len / query_len | 长度比 |
| 10 | len_diff | \|len1 - len2\| | 绝对长度差 |
| 11 | query_len | query 的字符数 | Query 本身长度 |
| 12 | bm25_score | 轻量 BM25 分数 | 经典 IR 相关性分数 |

> BM25 分数在特征重要性中排名第 2，印证了传统 IR 特征在排序中仍然不可或缺。

### IR 两阶段检索

```
段落库 (958,846篇) ──→ TF-IDF 向量化 ──→ 余弦相似度 ──→ Top-200 ──→ LR Rerank ──→ Top-10
```

1. **第一阶段（初筛）**：将 96 万段落用 TfidfVectorizer 向量化为稀疏矩阵，对每个 query 通过稀疏矩阵乘法计算余弦相似度，取出 top-200
2. **第二阶段（Rerank）**：对 top-200 候选构建特征向量（12 维手动 + TF-IDF），LR 模型预测 P(label≥2) 作为排序分，输出 top-10

> 初筛的 TF-IDF 矩阵和向量化器会缓存到 `models/ir_cache/`，下次运行直接加载。
>
> **全量评估**：`infer_ir.py` 会对每个 query 的初筛 top-200 和 rerank 后的 top-200 完整候选集合分别统计 NDCG@10、NDCG@200（不截断）、MRR、Recall@10 / Recall@200，并输出初筛 → rerank 的指标提升量，用数据直接验证重排的有效性。

---

## 环境要求

```
Python >= 3.8
scikit-learn, pandas, numpy, jieba, python-Levenshtein,
matplotlib, seaborn, joblib, scipy
```

安装依赖：

```bash
pip install scikit-learn pandas numpy jieba python-Levenshtein matplotlib seaborn joblib scipy
```

jieba 如遇构建问题，可使用：

```bash
pip install --no-build-isolation jieba
```

---

## 运行指南

所有脚本均从项目根目录运行。路径已基于 `__file__` 动态解析，无需手动修改。

### 0. 一键运行全部步骤（推荐）

```bash
python run_pipeline.py                     # 依次执行 数据探查 → 训练 → QTR推理 → IR检索
python run_pipeline.py --step train        # 仅执行某一步骤
```

可用 `--step` 值：`data_prepare` / `train` / `infer_qtr` / `infer_ir` / `all`（默认）。

### 1. 数据探查

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.data_prepare import main; main()"
```

### 2. 训练 + 评估

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.train_eval import main; main()"
```

训练过程包含：
- **GroupKFold (3折)** 交叉验证（按 query 分组，防止数据泄露）
- **GridSearchCV** 超参数搜索：TF-IDF max_features ∈ {2000, 4000}，LR C ∈ {0.5, 1.0}，共 16 组合 × 3 折 = 48 次训练
- 验证集评估（Accuracy, NDCG@10, 全量 NDCG, MRR）
- 混淆矩阵与特征重要性图生成
- 模型保存到 `models/lr_rerank.pkl`

### 3. KUAKE-QTR 测试集推理

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.infer_qtr import main; main()"
```

输出：`output/KUAKE-QTR_test_pred.json`（5,465 条，含 predict_label 和 predict_score）

### 4. KUAKE-IR 检索 + Rerank

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.infer_ir import main; main()"
```

输出：`output/KUAKE-IR_dev_pred.tsv`（1,000 query × 10 doc，共 10,000 行），并在终端打印初筛 vs rerank 的对比评估。

> ⚠ IR 全量检索需加载 96 万段落构建 TF-IDF 索引，占用内存约 2-4 GB，首次运行耗时约 2-3 分钟，后续加载缓存仅需数秒。

---

## 评估结果

### KUAKE-QTR（验证集）

| 指标 | 数值 | 含义 |
|------|------|------|
| Accuracy | 0.5613 | 4 分类准确率（随机基线约 25%） |
| NDCG@10 | **0.8689** | 排序质量高 |
| MRR | 0.6152 | 首个相关文档平均排在第 2 位 |

![KUAKE-QTR 混淆矩阵](./output/confusion_matrix.png)

### KUAKE-IR（验证集，1,000 query）

IR 评估基于**全量候选集合**（初筛 top-200）统计，并**对比初筛与 rerank** 两阶段指标，以验证重排有效性：

| 评估项 | 初筛 (TF-IDF top-200) | Rerank (LR top-200) | 说明 |
|--------|----------------------|---------------------|------|
| NDCG@10 | 0.20 | 0.29 | 全量排序后截断 top-10 计算 |
| NDCG@200（不截断） | — | 0.06 | 完整候选列表的全量 NDCG |
| MRR | 0.07 | 0.08 | 首个相关 doc 的平均位置倒数 |
| Recall@10 | 0.02 | 0.03 | top-10 命中率 |
| Recall@200 | 低 | 保持 | 相关 doc 是否进入初筛候选 |

> 说明：IR 每个 query 仅标注 1 个相关 doc，相关标注稀疏，NDCG/MRR 绝对值偏低属预期。`infer_ir.py` 会在运行后打印上述**初筛 → rerank** 的精确对比与提升量，请以实际运行输出为准。注：以上为全量统计框架示意，具体数值请运行 `infer_ir.py` 获取。

### 最佳超参数

| 参数 | 最佳值 |
|------|--------|
| TF-IDF max_features（query） | 2000 |
| TF-IDF max_features（title） | 2000 |
| TF-IDF max_features（concat） | 4000 |
| LR C（正则化强度倒数） | 0.5 |

### 特征重要性 Top-5（以实际训练输出为准）

| 排序 | 特征 | 权重 |
|------|------|------|
| 1 | concat TF-IDF 特征 | ~1.2 |
| 2 | **BM25 分数** | **~1.2** |
| 3-5 | TF-IDF 特征 | ~0.9~1.0 |

![特征重要性 Top-30](./output/feature_importance.png)

---

## 模型使用示例

```python
import joblib
import pandas as pd

# 加载模型
pipeline = joblib.load("models/lr_rerank.pkl")

# 单条预测
df = pd.DataFrame([{
    "query": "糖尿病饮食注意事项",
    "title": "糖尿病患者饮食指南",
    "label": -1  # dummy
}])
probs = pipeline.predict_proba(df)
score = probs[0, 2] + probs[0, 3]  # P(label≥2) 作为排序分
pred = pipeline.predict(df)[0]

print(f"预测标签: {pred}, 排序分数: {score:.4f}")
```

---

## 产出物

| 文件 | 格式 | 说明 |
|------|------|------|
| `models/lr_rerank.pkl` | pickle | 完整 sklearn Pipeline，可加载推理（1.7 GB，LFS 跟踪） |
| `output/KUAKE-QTR_test_pred.json` | JSON | 5,465 条测试预测 |
| `output/KUAKE-IR_dev_pred.tsv` | TSV | 检索排序结果 |
| `output/eval_report.txt` | Text | 完整评估报告 |
| `output/confusion_matrix.png` | PNG | 混淆矩阵热图 |
| `output/feature_importance.png` | PNG | 特征重要性 Top-30 |
| **`lr_rerank_report.html`** | **HTML** | **全流程技术报告（含图文 SVG 图解、概念通俗讲解）** |

---

## 如何进一步优化

- **Pairwise 训练**：使用 LambdaRank 直接优化排序指标，替代 Pointwise 分类
- **语义特征**：引入预训练语言模型（如 BERT）提取深层语义特征替代 TF-IDF
- **领域特征**：增加医学实体匹配、同义词扩展等医学领域特征
- **更大的候选集**：初筛从 top-200 扩展到 top-500 或 top-1000

## 参考

- [CBLUE 基准](https://github.com/aliyun/research-duty) - 中文生物医学语言理解评估
- [scikit-learn Pipeline & FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [全流程图文报告](./lr_rerank_report.html) - 含 SVG 图解的完整技术文档

---

> **Apache 2.0 License**
