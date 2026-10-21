"""Build the Chinese delivery report from measured, complete result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
full = json.loads((ROOT / "results/full_test/metrics.json").read_text(encoding="utf-8"))
small = json.loads((ROOT / "results/small_test/metrics.json").read_text(encoding="utf-8"))
a = full["metrics"]["legacy_dense_rerank"]
b = full["metrics"]["hybrid_rrf_rerank"]
c = full["metrics"]["bm25_rerank"]
lo, hi = full["comparison"]["cluster_bootstrap_95_ci"]
delta = b["recall_at_5"] - a["recall_at_5"]
relative = delta / a["recall_at_5"]
lines = [
    "# 油气公开数据：混合检索对照实验报告",
    "",
    f"本次主实验中，已标注证据 Recall@5 从 **{a['recall_at_5']:.2%}** 提升到 "
    f"**{b['recall_at_5']:.2%}**，增加 **{delta*100:.2f} 个百分点**，相对提升 **{relative:.2%}**。",
    "这是一轮公开英文语料、小样本、AI 核对标注的试验，不是中文客户生产系统指标。",
    "",
    "## 已完成内容",
    "",
    "- 原纯向量入口已封存为 `legacy_embeddings.py`，用 SHA-256 校验后运行。",
    "- 新增 `backend/HybridRetrieval.py`：BM25、向量召回、RRF 融合与既有 Cohere 重排接口。",
    "- 下载固定版本 DISKOS-QA，整理全量公开问答、去重语料、评测标注及可独立查看的小数据集。",
    "- 完成主实验、小库复测、BM25 + 同款重排的补充对照；保存逐题排名、指标与失败案例。",
    "",
    "## 数据集",
    "",
    "来源：[DISKOS-QA 原作者项目](https://github.com/georgeghon/DISKOS-QA)、"
    "[固定版本下载来源](https://huggingface.co/datasets/porestar/diskos_qa/tree/627ecd293877d943aac18914c0ce86053d6c1b97)。",
    "源项目声明 NLOD 2.0，上游 FORCE 语料声明 CC BY 4.0；署名、许可链接与派生说明见 README。",
    "",
    "| 项目 | 数量 / 规则 |",
    "|---|---|",
    "| 原始公开问答 | 1,027 条；不是全部都有可靠相关性标注 |",
    "| 去重原文语料 | 1,628 条片段，只索引原文，不把问答写入检索库 |",
    "| 初始固定样本 | 8 个主题各 5 题，按固定哈希排序选取，共 40 题 |",
    "| 排除 | 2 题；地层混用、深度自相矛盾；检索前排除并保留理由 |",
    "| 保留样本 | 8 道开发样例 + 30 道测试题 |",
    "| 小语料 | 171 条片段，包括保留问题的原始上下文干扰项 |",
    "",
    "原始数据的多个 context 不都是正确答案依据。本次在其上下文池内逐题做了 AI 证据核对，"
    "但未做全库穷尽标注，也未经领域专家验收。指标分母是这些已标注证据。",
    "抽样、标签和参数在第一次检索前冻结；没有根据测试结果选择融合权重或挑选获胜问题。",
    "",
    "## 对照配置",
    "",
    "| 项目 | 原始检索 | 混合检索 |",
    "|---|---|---|",
    "| 编码与向量库 | 默认 ONNX all-MiniLM-L6-v2 + Chroma L2 | 相同 |",
    "| 候选 | 向量 Top100 | 向量 Top50 + BM25 Top50 |",
    "| 融合 | 无 | RRF，常数 60，去重后最多 100 条 |",
    "| 重排 | rerank-english-v3.0 | 相同 |",
    "| 最终证据 | Top5，分数不低于 0.01 | 相同 |",
    "",
    "基线直接执行原类的 `retrieve_documents` 方法，仅绕过带旧机器路径的构造器，注入独立实验索引。"
    "两个分支使用完全相同的原始发布片段。没有重跑原上传器的 1,000 字符切分；"
    "本机 Chroma 为 0.6.3，旧 requirements 为 0.4.24，所以这不是历史线上环境复刻。",
    "",
    "## 主实验：1,628 条语料，30 道测试题",
    "",
    "| 方法 | 候选 Recall@100 | 最终 Recall@5 | Hit@5 | MRR@5 | NDCG@5 |",
    "|---|---:|---:|---:|---:|---:|"
]
names = {"legacy_dense_rerank":"原始向量 + 重排", "hybrid_rrf_rerank":"混合检索 + 重排",
         "bm25_only":"BM25，不重排（诊断）", "bm25_rerank":"BM25 + 同款重排（补充对照）"}
for method in names:
    s = full["metrics"][method]
    lines.append(f"| {names[method]} | {s['candidate_recall_at_100']:.2%} | {s['recall_at_5']:.2%} | "
                 f"{s['hit_at_5']:.2%} | {s['mrr_at_5']:.4f} | {s['ndcg_at_5']:.4f} |")
cmp = full["comparison"]
lines += [
    "",
    f"逐题改善 {cmp['improved_queries']} 道、退步 {cmp['regressed_queries']} 道、持平 {cmp['unchanged_queries']} 道。"
    f"按共享证据聚类的配对 bootstrap 95% 区间为 **[{lo*100:.2f}, {hi*100:.2f}] 个百分点**。"
    "这个区间只描述小样本在当前标签下的统计波动，不覆盖标注错误、语料偏差或中文迁移风险。",
    "",
    f"混合与 BM25 + 相同重排相比，Recall@5 差值为 **{(b['recall_at_5']-c['recall_at_5'])*100:+.2f} 个百分点**。"
]
if abs(b["recall_at_5"] - c["recall_at_5"]) < 1e-9:
    lines.append("因此本轮支持“补充词法召回改善原始向量方案”，但**没有证明融合优于单路 BM25 + 重排**。")
elif b["recall_at_5"] < c["recall_at_5"]:
    lines.append("因此本轮虽优于原始向量方案，但**混合方案不如 BM25 + 重排**；不能只展示有利对照。")
else:
    lines.append("混合方案相对 BM25 + 重排有点估计增益；该对照为主实验后追加诊断，仍需更大未见集验证。")
lines += [
    "这批题含井号、地层名、专业缩写，词法匹配优势很强。补充对照是在看到 BM25-only 表现后追加，未调参。",
    "",
    "## 小库复测：171 条语料，相同 30 道测试题",
    "",
    "| 方法 | Recall@5 | Hit@5 |",
    "|---|---:|---:|"
]
for method in ["legacy_dense_rerank","hybrid_rrf_rerank"]:
    s = small["metrics"][method]
    lines.append(f"| {names[method]} | {s['recall_at_5']:.2%} | {s['hit_at_5']:.2%} |")
slo, shi = small["comparison"]["cluster_bootstrap_95_ci"]
lines += [
    "",
    f"提升 {small['comparison']['mean_delta']*100:.2f} 个百分点，仅 {small['comparison']['improved_queries']} 题改善。"
    f"95% 区间 [{slo*100:.2f}, {shi*100:.2f}] 个百分点，包含零。"
    "小库更容易出现满分，不能把它当作大规模知识库的性能。",
    "",
    "## 效果来自哪里",
    "",
    "| 题目类型 | 原始方案的问题 | 混合方案 |",
    "|---|---|---|",
    "| 1/6-6 井的气测范围 | 检索到类似描述但属于其他井；标注证据不在候选集 | 补回对应井的原文片段 |",
    "| Lunde 地层净砂截止值 | 返回相似但属于其他地层的截止值讨论 | 补回 Lunde 证据 |",
    "| Y2H 的 M-sand/C-sand 原因 | 最终证据为空，标注证据未进候选 | 补回解释更深储层位置的片段 |",
    "| Lysing 地层多井信息 | 少命中一条预标注证据 | 覆盖两条预标注证据 |",
    "",
    "第四题原始结果中还有未标注的潜在替代证据，见 `posthoc_audit.json`。"
    "因此已标注片段未命中，不必然等于无法回答；不能把 Recall 增益直接改写为答案准确率增益。",
    "",
    "另一个重要限制：1,628 条片段中有 1,307 条超过默认编码器的 256-token 截断长度。"
    "BM25 读取全文，可能弥补了向量只看前文的损失。"
    "应另行用短分块或更合适的长文本嵌入作为强基线，才能判断剩余的混合检索收益。",
    "",
    "## 时延与资源",
    "",
    f"- 原始方案：P50 {a['p50_seconds']:.3f}s，P95 {a['p95_seconds']:.3f}s。",
    f"- 混合方案：P50 {b['p50_seconds']:.3f}s，P95 {b['p95_seconds']:.3f}s。",
    f"- 平均重排候选：原始 {a['candidate_count']:.1f}，混合 {b['candidate_count']:.1f}。",
    "- 每组 30 次串行调用；排除了主动限速等待，含网络与服务端波动；不是并发压测。",
    "- CPU 嵌入运行于 i7-14650HX、约 16 GB 内存机器，重排调用外部 Cohere 服务。",
    "",
    "## 应怎样描述成果",
    "",
    f"可以描述为：“在公开英文油气报告的 30 题试验集上，"
    f"将原始向量召回扩展为 BM25/向量混合检索，已标注证据 Recall@5 从 {a['recall_at_5']:.2%} "
    f"提高到 {b['recall_at_5']:.2%}；完成同模型对照及逐题分析。”",
    "",
    "目前不宜描述为中石油业务数据的线上提升、专家人工评测结果、中文系统效果，"
    "也不宜宣称混合检索已经胜过所有单路方案。",
    "",
    "下一步应优先补充领域专家盲审、全方法候选池标注、口语与同义表达问题，"
    "以及按来源文档分组的较大未见测试集；当前测试集今后只作回归集。",
    "",
    "## 文件入口",
    "",
    "- `data/small_dataset.json`：可独立查看的小数据集。",
    "- `data/corpus.jsonl` / `questions_all.jsonl`：整理后的完整公开片段与问答。",
    "- `data/queries_test.jsonl` / `annotations.json`：测试题及证据标注理由。",
    "- `results/full_test/per_query.csv`：每题、每种方法的具体指标。",
    "- `results/full_test/predictions.jsonl`：原始候选和最终排名。",
    "- `README.md`：许可、完整限制、复现命令及业务接入方式。",
    "",
    "原始检索文件校验通过；算法、指标、权限过滤与数据完整性测试通过。"
]
(ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(ROOT / "RESULTS_SUMMARY.md")
