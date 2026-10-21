"""Create an auditable report from measured results, never from target scores."""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from evaluate import read_jsonl, aggregate, paired

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
NAMES = {"page_dense": "整页向量（截断诊断）", "legacy1000_dense": "原 1,000 字符分块＋向量",
         "token_dense": "适配 token 分块＋向量", "bm25": "适配分块＋BM25",
         "hybrid": "适配分块＋混合 RRF", "ungated_hybrid": "未过滤混合对照"}


def pct(value):
    return f"{100*value:.2f}%"


def table(aggregates, rerank=False):
    lines = ["| 方法 | 题数 | 证据覆盖 Recall@5 | 全证据齐全@5 |" + ("" if rerank else " 候选 Recall@100 |"),
             "|---|---:|---:|---:|" + ("" if rerank else "---:|")]
    for m in ["page_dense", "legacy1000_dense", "token_dense", "bm25", "hybrid"]:
        if m not in aggregates:
            continue
        a = aggregates[m]
        lines.append(f"| {NAMES[m]} | {a['n']} | {pct(a['evidence_recall@5'])} | {pct(a['all_evidence@5'])} |" +
                     ("" if rerank else f" {pct(a['evidence_recall@100'])} |"))
    return "\n".join(lines)


def main():
    result = json.loads((ROOT / "results/metrics.json").read_text())
    rerank = json.loads((ROOT / "results/rerank_metrics.json").read_text())
    diag = json.loads((ROOT / "results/index_diagnostics.json").read_text())
    qs = {q["id"]:q for q in read_jsonl(ROOT / "data/queries.jsonl")}
    docs = {d["id"]:d for d in read_jsonl(ROOT / "data/documents.jsonl")}
    per_query = read_jsonl(ROOT / "results/per_query_metrics.jsonl")
    public = result["aggregates"]["test/public_ocr"]
    delta = (public["hybrid"]["evidence_recall@5"] - public["token_dense"]["evidence_recall@5"])*100
    sparse_delta = (public["hybrid"]["evidence_recall@5"] - public["bm25"]["evidence_recall@5"])*100
    text = [
        "# 油气检索评测 v2：实测结果与适用范围",
        "",
        f"在 112 道公开资料测试题上，混合检索相对适配分块后的纯向量，"
        f"前 5 块已标注证据覆盖率提高 **{delta:.2f} 个百分点**；"
        f"相对 BM25 单路为 **{sparse_delta:+.2f} 个百分点**。"
        "因此本轮支持“混合优于当前纯向量基线”，不支持“混合最优”。",
        "",
        "这里评估的是原文证据覆盖，不是回答准确率。全部结果来自本地索引与实际 API "
        "调用；没有按目标分数调整题目。未经专家确认的金标、OCR 与出题方式仍会影响结果。",
        "",
        "## 数据规模",
        "",
        "- 40 份公开油气报告的 758 个可用 OCR 导出页；8 份报告有页码缺口。",
        "- 20 份报告出题：60 个基础事实、直接问法与改写、20 个两部分问题，共 140 题。",
        "- 50 条模拟行政归档记录、80 个功能边界问题，与公开题分开。",
        "- 总计 220 题；开发 44、测试 176。测试公开题 112，来自 16 份来源报告。",
        "- 题目与源文字符证据在看分数前冻结；按来源系列分组。相同事实改写不是独立样本。",
        "- 来源：[FORCE / Zenodo](https://zenodo.org/records/10775273)。英语公开资料，"
        "不等同于中石油业务数据、中文评测或客户使用日志。",
        "",
        "## 第一阶段检索：公开测试集",
        "",
        table(public),
        "",
        "前 5 块与前 100 候选是不同指标。混合候选为向量 50＋BM25 50 后去重，"
        "单路最多 100；混合并不包含单路的全部 Top100。",
        "",
        "同一原文位置的完整证据被覆盖才记分，可合并相邻块覆盖；同一文档内命中其他位置"
        "不算正确。这个定义比“命中任意相关页”严格，但由于未穷尽语义等价证据，"
        "应称为已标注证据覆盖率，不能包装成真实线上绝对召回率。",
        "",
        "## 相同重排器：预先固定的 32 题挑战子集",
        "",
        table(rerank["aggregates"], True),
        "",
        "每个测试报告选 1 个改写题和 1 个多证据题。三路候选分别使用同一个 "
        "Cohere rerank-english-v3.0，Top5、阈值 0.01。"
        "这张表只能在 32 题内横向比较，不能与上面的 112 题混算。",
        "",
        "## 配对不确定性",
        ""
    ]
    for c in result["paired_comparisons"]:
        if c["metric"] != "evidence_recall@5":
            continue
        lo, hi = c["family_bootstrap_95ci_pp"]
        text.append(f"- {NAMES[c['challenger']]} 相对 {NAMES[c['baseline']]}："
                    f"{c['delta_pp']:+.2f} pp，来源组 bootstrap 95% 区间 [{lo:.2f}, {hi:.2f}] pp；"
                    f"{c['improved']} 改善、{c['regressed']} 退步、{c['unchanged']} 不变。")
    text += ["", "重排子集配对结果：", ""]
    for c in rerank["paired"]:
        lo, hi = c["family_bootstrap_95ci_pp"]
        text.append(f"- 混合相对 {NAMES[c['baseline']]}：{c['delta_pp']:+.2f} pp，"
                    f"95% 区间 [{lo:.2f}, {hi:.2f}] pp，{c['n']} 题 / "
                    f"{c['independent_source_families']} 个来源组。")
    text += ["", "区间只描述当前来源样本的波动，不消除出题偏差或金标遗漏。",
             "", "## 分场景结果", ""]
    for category, label in [("direct", "直接问法"), ("paraphrase", "业务化改写"),
                            ("multi_evidence", "两部分／多证据")]:
        text += [f"### {label}", "", table(result["aggregates"][f"test/public_ocr/{category}"]), ""]
    text += ["## 分块与向量化", ""]
    for kind, label in [("page", "整页"), ("legacy1000", "原 1,000 字符"), ("token", "适配 token")]:
        d = diag[kind]
        text.append(f"- {label}：{d['chunks']} 块，超过 256 token 的 {d['over_256']} 块，"
                    f"最大 {d['max_tokens']} token。")
    text += ["", "模型与原项目一致：all-MiniLM-L6-v2，384 维。适配分块把标题和特殊 token "
             "计入上限，重叠 32 token。向量实际生成并缓存在本地；检索用精确归一化点积，"
             "与归一化向量 L2 排序等价。不是只用 BM25 假装混合检索。",
             "", "分块对照改变了长度、重叠和候选粒度，因此不能把全部变化单独归因于“消除截断”。"
             "不同大小块的 TopK 文本预算也不同；最公平的混合与单路比较使用相同 token 分块。",
             "", "## 权限、时间和澄清功能检查", "",
             "| 方法 | 测试题 | Top5 含越权内容的题数 | Top5 含错误有效期内容的题数 | 触发站点澄清 |",
             "|---|---:|---:|---:|---:|"]
    for m in ["ungated_hybrid", "hybrid"]:
        s = result["policy_checks"][m]
        text.append(f"| {NAMES[m]} | {s['n']} | {s['queries_exposing_unauthorized_top5']} | "
                    f"{s['queries_exposing_wrong_date_top5']} | {s['clarifications_triggered']} |")
    text += ["", "这是显式元数据条件下的模拟功能检查。允许角色、站点和 as-of 日期是输入，"
             "未对接真实登录态，未实现自然语言日期解析。40 道有答案的模拟测试题各方法"
             "均为 100% 覆盖，说明模板检查容易，不能把它们并入公开题来提高总分。",
             "", "8 道库内无答案测试题仍全部返回了候选。检索到文本不等于能回答；"
             "本轮没有回答生成或可靠拒答器，所以不报告拒答准确率。",
             "", "## 事后审阅与敏感性分析", "",
             "冻结后发现 `public-f053-paraphrase` 使用缺少前文的“那两个机会”。"
             "原始主表保留，下面单列剔除这一争议题后的敏感性结果，"
             "不是预先注册的主结果：", ""]
    clean_rows = [r for r in per_query if r["split"] == "test" and r["source_type"] == "public_ocr"
                  and r["query_id"] != "public-f053-paraphrase"]
    sensitivity = {m: aggregate([r for r in clean_rows if r["method"] == m]) for m in public}
    text += [table(sensitivity), "",
             "更多问题见 `posthoc_audit.json`：PL127B 编号形式不一致、"
             "同一油田对象跨报告/时间出现、50/50 候选预算漏掉单路尾部证据。"
             "没有静默修正金标或对测试集调融合权重。",
             "", "## 对改进方向与简历的建议", "",
             "优先补齐编号归一化和来源/时间约束，再在新的独立留出集上比较 BM25＋重排与混合＋重排。"
             "本轮已经看过测试结果，不应用同一套题调出更好分数后继续称为无偏测试。",
             "",
             f"当前可据实描述：构建 220 条油气领域公开与模拟评测问题，"
             f"实现 token 长度感知分块、BM25 与向量混合检索及消融对照；"
             f"在 112 条公开资料测试问题上，第一阶段 Evidence Recall@5 "
             f"由 {pct(public['token_dense']['evidence_recall@5'])} 提升至 "
             f"{pct(public['hybrid']['evidence_recall@5'])}（+{delta:.2f} pp），"
             "同时识别出 BM25 单路更优的场景。",
             "",
             "这些是本地复现实验的结果。是否归入实习职责，应与实际承担工作一致；"
             "不能写成客户已上线的效果、专家验收结论或真实生产召回率。",
             "", "## 复现与审计入口", "",
             "- `README.md`：来源、选择规则、执行命令、环境与局限。",
             "- `facts.tsv` / `data/queries.jsonl`：问题和证据。",
             "- `data/frozen_hashes.json`：冻结输入与封存纯向量基线的哈希。",
             "- `results/local_rankings.jsonl` / `rerank_rankings.jsonl`：逐题原始输出。",
             "- `results/metrics.json` / `rerank_metrics.json`：完整统计与配对区间。",
             "- `FAILURE_CASES.md`：可逐项核对的失败案例。",
             "- `results/environment.json`：版本、脚本和结果文件哈希。",
             ""]
    (ROOT / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    (ROOT / "results/posthoc_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    # Stable representative cases, preserved regardless of reranking outcome.
    chunks = {c["id"]: c for c in read_jsonl(ROOT / "data/chunks_token.jsonl")}
    ranks = {r["query_id"]:r for r in read_jsonl(ROOT / "results/local_rankings.jsonl")}
    cases = ["# 失败案例审阅", "", "以下为冻结主实验中的案例，不据此修改主标签。", ""]
    for qid in ["public-f009-direct", "public-f023-direct", "public-f053-paraphrase", "public-f056-paraphrase", "public-04-multi"]:
        q = qs[qid]
        cases += [f"## {qid}", "", q["query"], "", "标注证据：", ""]
        for group, quote in zip(q["evidence_groups"], q["reference_evidence"]):
            span = group[0]
            d = docs[span["document_id"]]
            cases += [f"- {d['title']}，导出页 {d['page']}，字符 {span['start']}–{span['end']}：{quote}"]
        for method in ["token_dense", "bm25", "hybrid"]:
            r = next(r for r in per_query if r["query_id"] == qid and r["method"] == method)
            cases += ["", f"### {NAMES[method]}",
                      "", f"Recall@5 = {pct(r['metrics']['evidence_recall@5'])}；"
                      f"Recall@100 = {pct(r['metrics']['evidence_recall@100'])}。", ""]
            for cid in ranks[qid]["rankings"][method][:3]:
                c = chunks[cid]
                cases += [f"- `{cid}`（源字符 {c['start']}–{c['end']}）："
                          + c["text"][:650].replace("\n", " ") + "…"]
    (ROOT / "FAILURE_CASES.md").write_text("\n".join(cases)+"\n", encoding="utf-8")
    versions = {}
    for package in ["numpy", "chromadb", "onnxruntime", "tokenizers", "cohere", "langchain-text-splitters", "requests"]:
        versions[package] = importlib.metadata.version(package)
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.tsv")) + list(ROOT.glob("*.json"))
    paths += list((ROOT / "results").glob("*.jsonl")) + list((ROOT / "data").glob("*.jsonl"))
    paths += [PROJECT / "backend/Embeddings.py", PROJECT / "backend/HybridRetrieval.py",
              PROJECT / "backend/TokenChunking.py",
              ROOT.parent / "petroleum_retrieval/legacy_embeddings.py"]
    env = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
           "files_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           "test_commands": ["python -m unittest discover -s experiments/production_proxy -p test_protocol.py",
                             "python -m unittest discover -s experiments/petroleum_retrieval -p test_retrieval.py"]}
    (ROOT / "results/environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE RESULTS.md, FAILURE_CASES.md, sensitivity, environment", flush=True)


if __name__ == "__main__":
    main()
