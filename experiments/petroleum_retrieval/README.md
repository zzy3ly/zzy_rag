# 公开油气语料：原始检索与混合检索对照实验

本目录用于验证“BM25 + 向量混合召回是否优于现有单路向量召回”，不预设一定提升。
原 `backend/Embeddings.py`、业务数据库、现有 Chroma 索引和聊天接口均不改动。
新实现为 `backend/HybridRetrieval.py`，由本目录脚本实际调用。

## 数据来源与许可

- DISKOS-QA 原作者项目：<https://github.com/georgeghon/DISKOS-QA>
- 下载镜像：<https://huggingface.co/datasets/porestar/diskos_qa>
- 固定 revision：`627ecd293877d943aac18914c0ce86053d6c1b97`
- 原始 FORCE 语料：FORCE NETWORK Group (2024),
  *Large Oil and Gas industry text dataset from Norwegian, UK and Dutch public oil and gas documents*,
  DOI <https://doi.org/10.5281/zenodo.10775273>
- DISKOS-QA 声明 NLOD 2.0：<https://data.norge.no/nlod/en/2.0>；
  数据卡说明上游 FORCE 语料为 CC BY 4.0：<https://creativecommons.org/licenses/by/4.0/>。
- 本地变换：空白规范化、按文本哈希精确去重、固定抽样、AI 核对相关证据、排除矛盾样例。
  保留原始来源 chunk ID、文档 ID 和文件名；未修改原始 CSV。
- 发布本派生数据时必须保留本说明、原作者署名、许可链接和变更说明。
  原始公开历史报告中的 `CONFIDENTIAL` 等字样不表示本项目接触了客户内部文件；
  此处材料来自上述公开发布数据集，不代表中石油授权或客户交付。

这是英文为主的公开地下地质/钻完井/测井资料，不是中文设备运维规程，
不能把实验结果直接写成中文企业内部知识库或线上业务效果。

## 为什么选择它

| 数据集 | 本轮适用性 |
|---|---|
| DISKOS-QA | 有问题、答案、原文片段与来源 ID，能构建检索语料；但上下文列表不是逐段相关性标注 |
| WellQuest | 有钻井日报和关联行标注，领域合适；本次公开下载 API 返回 403，未使用其数据 |
| PetroBench | 中英文专业题目可用于知识问答；已查看的题目不提供本轮所需的原文证据正文，未将答案伪装成检索语料 |
| FormationEval | 主要为专业选择题，不能直接替代带证据的检索评测 |

参考：
<https://data.mendeley.com/datasets/f64ny7mcvz/1>
<https://huggingface.co/datasets/my2000cup/PetroBench>
<https://github.com/AlmazErmilov/FormationEval-an-Open-Benchmark-for-Oil-Gas-Geoscience-MCQ-Evaluation>

## 数据规模与标注

- 原始 1,027 条问答，1,690 个来源 chunk ID。
- 规范化空白并精确去重后：1,628 个原文片段，索引不含问题或标准答案。
- 从 `Entailed=True` 的问题中，按 8 个主题各抽 5 条，共 40 条。
- 以 `SHA256("petroleum-v1-20261020:" + Question_ID)` 排序；每主题第 1 条为开发样例，其余为测试。
- 检索运行前，AI 核对原始候选上下文；排除 2 条存在地层混用或深度矛盾的问题。
- 最终 8 条开发样例、30 条测试问题。没有按算法胜负筛选，也没有在测试集调参。
- `annotations.json` 保存每题相关片段选择与理由。
- `data/excluded.jsonl` 保存所有排除项及理由。
- `data/corpus_small.jsonl` 是 38 条保留问题原始上下文并集，171 条片段，保留了上下文干扰项。
- 主试验用 **完整 1,628 条片段 + 30 条小测试集**。
- 小库补充试验用 **171 条片段 + 相同 30 条测试问题**，不能当作全库结论。

标注限制：

1. 原问答主要由 LLM 辅助生成；本次是 AI 核对，不是领域专家或人工金标。
2. 仅在上游提供的上下文池中判相关性，没有对全库进行穷尽标注。
   库内可能还有未标注但同样相关的片段；因此是“相对已标注证据的 Recall”。
3. 保留原文已有 OCR 缺损。评测证据检索，不证明答案数值在工程上正确。
4. 有重叠意图题；置信区间按共享已标注证据聚类。其他文档层面相关性仍可能存在。
5. 开发集与测试集没有严格按来源文档隔离，但本轮没有训练模型或开发集调参。
   后续调参必须建立按来源文档分组的新留出集，不能继续把本测试集当成未见集。
6. 两组使用原始发布片段，而非重新运行旧上传器的 1,000 字符切分。
   这是旧检索方法在共享新语料上的实验，不是历史系统线上成绩的复测。
7. 默认 MiniLM 编码器截断为 256 token；1,628 条片段中 1,307 条超过此长度。
   BM25 和重排读取完整原文，因此混合检索的收益可能包含弥补向量编码截断的贡献。
   要验证优化后的短分块/长上下文嵌入基线，必须另做控制实验；
   不能据本轮声称对任意强向量基线都有同样收益。

## 固定实验配置

| 项目 | 原始方法 | 混合方法 |
|---|---|---|
| 嵌入 | Chroma 默认 ONNX all-MiniLM-L6-v2 | 相同 |
| 向量库 | 独立实验 Chroma，L2，search_ef=100 | 同一索引 |
| 向量候选 | Top100 | Top50 |
| 关键词候选 | 无 | BM25 Top50 |
| 融合 | 无 | 等权 RRF，常数 60 |
| BM25 | 无 | k1=1.5，b=0.75，保留井号/型号复合 token |
| 重排 | Cohere rerank-english-v3.0 | 相同 |
| 最终结果 | Top5，分数 >= 0.01 | 相同 |

候选上限相同，但混合去重后往往少于 100 条，报告实际候选数。
另输出 BM25-only 作为第一阶段诊断，它没有重排，不能冒充同设置的完整链路比较。

为了保留旧实现，脚本通过 `EmbeddingHandler.__new__` 注入实验 collection 和
带缓存的 Cohere 客户端，只绕过构造器中的旧机器路径及上传器初始化。
实际执行的是原文件的 `retrieve_documents` 方法，没有重写基线逻辑。
`data/frozen_hashes.json` 记录原文件 SHA-256，运行前校验。

运行环境使用本机已安装的 Chroma 0.6.3；旧 requirements 声明的是 0.4.24。
因此可说“同一环境、同一模型、原方法对照”，不可说完全复原了历史依赖环境。
实际依赖版本记录在每次的 `run_manifest.json`。

## 运行

在项目根目录执行：

```powershell
python -X utf8 experiments/petroleum_retrieval/prepare.py
python -X utf8 experiments/petroleum_retrieval/freeze_labels.py
python -X utf8 -m unittest discover -s experiments/petroleum_retrieval -p test_retrieval.py -v
python -X utf8 experiments/petroleum_retrieval/run_experiment.py
python -X utf8 experiments/petroleum_retrieval/evaluate.py full_test
python -X utf8 experiments/petroleum_retrieval/run_experiment.py --small-corpus
python -X utf8 experiments/petroleum_retrieval/evaluate.py small_test
python -X utf8 experiments/petroleum_retrieval/run_bm25_control.py
python -X utf8 experiments/petroleum_retrieval/evaluate.py full_test
python -X utf8 experiments/petroleum_retrieval/build_summary.py
```

使用已有环境，无须重装旧 requirements。所需库是 `requests`、`numpy`、
`chromadb`、`cohere`，以及旧 Embeddings 模块导入所需的 LangChain/HTML 依赖。

认证优先读 `COHERE_API_KEY` 环境变量，否则读取已有 `backend/cohere.txt`。
密钥不写入本目录。Cohere 调用仅发送公开片段和对应问题；可能使用账户调用额度。
客户端默认将请求间隔限制为 6.3 秒，遇到 429/服务异常做有界重试。
每题结果立即落盘，重跑可继续未完成项。
缓存保持重排结果以减少重复调用；缓存运行不算新的在线时延样本。

原始检索日志可能较长，保存在实验结果目录，不输出到聊天窗口。

## 输出与指标

- `RESULTS_SUMMARY.md`：中文综合结论与效果解读。
- `data/small_dataset.json`：原文片段与问答合并的小数据集，方便独立查看。

- `results/full_test/REPORT.md`：完整语料上的主结果。
- `results/small_test/REPORT.md`：小语料补充结果。
- `metrics.json`：指标、逐主题结果与聚类 bootstrap 区间。
- `predictions.jsonl`：每题每种方法的候选 ID、最终 ID、耗时。
- `per_query.csv` / `per_query.jsonl`：逐题分数及问题文本。
- `run_manifest.json`：参数、版本、输入哈希和运行条件。
- `posthoc_audit.json`：主试验后的失败案例审阅；指出一题存在未标注的替代证据，
  不以结果为导向悄悄修改冻结标注。

`run_bm25_control.py` 是看到 BM25-only 表现较强后追加的诊断对照，
使用 BM25 Top100 + 完全相同的重排与阈值，没有据此调整混合参数。
它用于分辨“补充词法检索的收益”和“融合相比单路 BM25 的额外收益”。

Recall@5 是每题命中相关片段数除以已标注相关片段数，再对题目宏平均。
Hit@5 是是否至少命中一个；All-evidence@5 是是否覆盖全部已标注证据。
MRR@5 衡量第一条相关证据的位置；NDCG@5 衡量排序质量。
Candidate Recall@100 测重排前集合，混合集合可能不足 100。

百分比差用“百分点”；相对提升 = (新值 - 旧值) / 旧值。
若配对区间跨零，只能说本样本的点估计变化，不能宣称稳定显著提升。
单次串行 API 时延包含网络波动，排除主动限速等待；不是并发性能承诺。

## 在业务代码中试用

可在服务启动时从现有 collection 获取受管理的语料快照，构造
`HybridRetriever(collection, documents, existing_handler.co)`，
其 `retrieve_documents(query, num_docs, metadata_filters)` 与旧返回结构一致。
`documents` 每项需有 `id`、`text`、`metadata`，元数据保留 `filename`。
显式 metadata filter 会作用于两路候选，空授权范围不会回退为全库查询。
身份与权限范围必须由上游服务端确定，本模块不提供认证。

本轮没有将新方法默认为线上聊天入口：现有功能继续使用旧方法，避免未经验证切换。
内存 BM25 是静态小规模索引，新增/删除文档后应重建快照；
不是分布式搜索服务或完整的生产权限系统。
