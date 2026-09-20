# KUAKE 医学段落检索 LR Rerank

基于 sklearn LogisticRegression 实现的轻量化 Pointwise Rerank，在 CBLUE 基准的 KUAKE-QTR（查询-标题相关性）和 KUAKE-IR（医学段落检索）数据集上完成训练、评估与推理。

## 项目结构

```
├── KUAKE-QTR/                      # QTR 数据集
│   ├── KUAKE-QTR_train.json        # 训练集 (24,174 条)
│   ├── KUAKE-QTR_dev.json          # 验证集 (2,913 条)
│   └── KUAKE-QTR_test.json         # 测试集 (5,465 条)
├── KUAKE-IR/                       # IR 数据集
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
├── models/
│   ├── lr_rerank.pkl               # 训练好的 LR Pipeline
│   └── ir_cache/                   # IR TF-IDF 向量化缓存
│       ├── ir_tfidf_vectorizer.pkl
│       └── ir_tfidf_matrix.npz
├── output/
│   ├── eval_report.txt             # 完整评估报告
│   ├── confusion_matrix.png        # 混淆矩阵
│   ├── feature_importance.png      # 特征重要性 Top-30
│   ├── KUAKE-QTR_test_pred.json    # QTR 测试集预测 (5,465 条)
│   ├── KUAKE-IR_dev_pred.tsv       # IR 检索排序结果 (10,000 行)
│   └── lr_rerank_report.html       # 全流程技术报告 (含图文)
└── README.md                       # 本文档
```

## 数据集

| 数据集 | 用途 | 规模 | 说明 |
|--------|------|------|------|
| KUAKE-QTR | Query-Title 相关性分类 | 24,174 train / 2,913 dev / 5,465 test | 4 分类 (0~3) |
| KUAKE-IR | 段落检索 | 958,846 篇段落 / 1,000 条标注查询 | 两阶段检索 |

## 方法

### Pipeline 架构

特征提取通过 `sklearn FeatureUnion` 并行计算 4 组特征，拼接后送入 `LogisticRegression`（multinomial softmax）：

1. **手动特征 (12 维)**：词重叠率、Jaccard 系数、Query/Title 相互命中比、编辑距离、最长公共子串、数字匹配、长度特征、BM25 分数
2. **Query TF-IDF**：字符级 char_wb ngram(1,3)
3. **Title TF-IDF**：字符级 char_wb ngram(1,3)
4. **Concat TF-IDF**：字符级 char ngram(1,2)

### IR 两阶段检索

1. **第一阶段 (初筛)**：TF-IDF 余弦相似度，从 96 万段落中召回 top-200
2. **第二阶段 (Rerank)**：LR Pipeline 对候选重新打分，取 P(label≥2) 作为排序分，输出 top-10

## 环境要求

```
Python >= 3.8
scikit-learn
pandas
numpy
jieba
python-Levenshtein
matplotlib
seaborn
joblib
scipy
```

安装依赖：

```bash
pip install scikit-learn pandas numpy jieba python-Levenshtein matplotlib seaborn joblib scipy
```

jieba 如遇构建问题，可使用：

```bash
pip install --no-build-isolation jieba
```

## 运行指南

所有脚本均需从项目根目录运行。

### 1. 数据探查

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.data_prepare import main; main()"
```

### 2. 训练 + 评估

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.train_eval import main; main()"
```

训练过程包含：
- GroupKFold (3折) 交叉验证
- GridSearchCV 超参数搜索（TF-IDF max_features、LR C）
- 验证集评估（Accuracy、NDCG@10、MRR）
- 混淆矩阵与特征重要性图生成
- 模型保存

### 3. KUAKE-QTR 测试集推理

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.infer_qtr import main; main()"
```

输出：`output/KUAKE-QTR_test_pred.json`（5,465 条预测）

### 4. KUAKE-IR 检索 + Rerank

```bash
python -c "import sys; sys.path.insert(0, '.'); from src.infer_ir import main; main()"
```

输出：`output/KUAKE-IR_dev_pred.tsv`（1,000 query × 10 doc）

## 评估结果

### KUAKE-QTR

| 指标 | 数值 |
|------|------|
| Accuracy | 0.5613 |
| NDCG@10 | 0.8689 |
| MRR | 0.6152 |

### KUAKE-IR

| 指标 | 数值 |
|------|------|
| NDCG@10 | 0.7158 |
| MRR | 0.0176 |
| Recall@10 | 0.0280 |

## 模型使用示例

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load("models/lr_rerank.pkl")

# 单条预测
query = "糖尿病饮食注意事项"
title = "糖尿病患者饮食指南"

import pandas as pd
df = pd.DataFrame([{"query": query, "title": title, "label": -1}])
probs = pipeline.predict_proba(df)
score = probs[0, 2] + probs[0, 3]  # P(label≥2) 作为排序分
pred = pipeline.predict(df)[0]

print(f"预测标签: {pred}, 排序分数: {score:.4f}")
```

## 产出物

| 文件 | 格式 | 说明 |
|------|------|------|
| `models/lr_rerank.pkl` | pickle | 完整 sklearn Pipeline，可加载推理 |
| `output/KUAKE-QTR_test_pred.json` | JSON | 5,465 条测试预测 |
| `output/KUAKE-IR_dev_pred.tsv` | TSV | 检索排序结果 |
| `output/eval_report.txt` | Text | 完整评估报告 |
| `output/confusion_matrix.png` | PNG | 混淆矩阵热图 |
| `output/feature_importance.png` | PNG | 特征重要性 Top-30 |
| `output/lr_rerank_report.html` | HTML | 全流程技术报告（含图文） |

## 参考

- [CBLUE 基准](https://github.com/aliyun/research-duty) - 中文生物医学语言理解评估
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
