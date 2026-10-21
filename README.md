# zzy_rag

一个可复现的油气领域 RAG 检索实验项目。仓库保留原始 FastAPI + Chroma
问答原型，并新增长度感知分块、BM25、向量检索、RRF 混合检索、统一重排
和证据跨度评测。

## 实验结论

生产场景代理评测包含 220 道问题，其中 140 道来自公开油气报告，80 道为
明确标注的版本、权限、无答案和澄清模拟题。公开测试集包含 112 道问题：

| 检索方法 | Evidence Recall@5 | Candidate Recall@100 |
|---|---:|---:|
| 适配 token 分块 + 向量 | 45.09% | 83.48% |
| 适配分块 + BM25 | 76.79% | 96.43% |
| 适配分块 + RRF 混合检索 | 60.71% | 95.54% |

混合检索相对纯向量提升 15.62 个百分点，但没有超过 BM25。固定 32 道挑战题
使用相同 Cohere 重排器后，纯向量为 84.38%，BM25 与混合检索均为 93.75%。
因此本项目不把“混合检索一定最优”当作预设结论。

完整定义、置信区间和局限见
[生产代理实验报告](experiments/production_proxy/RESULTS.md)。

## 系统组成

```text
文档上传
  -> 文本提取与分块
  -> Chroma / MiniLM 向量索引
  -> Dense + BM25 候选召回
  -> RRF 融合
  -> Cohere Rerank
  -> LLM 生成与引用
```

- `backend/`：原 FastAPI RAG 原型及新增检索模块。
- `backend/TokenChunking.py`：按嵌入模型 token 上限切分并保留源字符位置。
- `backend/HybridRetrieval.py`：当前在线使用的 BM25、RRF 和统一权限范围过滤实现。
- `experiments/production_proxy/`：220 题生产场景代理评测。
- `experiments/petroleum_retrieval/`：较早的 DISKOS-QA 对照实验。
- `frontend/`：原型前端页面。

## 数据来源

主要实验使用 FORCE NETWORK Group 发布的公开油气报告 OCR 数据：

- [Large Oil and Gas industry text dataset](https://doi.org/10.5281/zenodo.10775273)
- 本实验使用 Norwegian Petroleum Directorate relinquishment reports 导出。
- 原始 CSV 不纳入 Git；`download_sources.py` 会从 Zenodo 下载并校验 MD5。
- 40 份报告中有 8 份存在导出页码缺口，结果不声称恢复了完整 PDF。

问题与证据由 AI 辅助构建并逐字校验，没有冒充石油领域专家金标，也没有
使用企业客户日志。指标是已标注证据覆盖率，不是线上答案准确率。

## 复现实验

实验使用 Python 3.13 运行。依赖环境可参考
`experiments/petroleum_retrieval/requirements-experiment.txt` 与
`backend/requirements.txt`。在项目根目录执行：

```powershell
D:\python\python.exe -X utf8 experiments/production_proxy/download_sources.py
D:\python\python.exe -X utf8 experiments/production_proxy/build_dataset.py
D:\python\python.exe -X utf8 experiments/production_proxy/run_rerank.py --select-only
D:\python\python.exe -X utf8 -m unittest discover -s experiments/production_proxy -p test_protocol.py
D:\python\python.exe -X utf8 experiments/production_proxy/run_local.py
D:\python\python.exe -X utf8 experiments/production_proxy/evaluate.py
D:\python\python.exe -X utf8 experiments/production_proxy/run_rerank.py
D:\python\python.exe -X utf8 experiments/production_proxy/build_report.py
```

重排实验优先从环境变量 `COHERE_API_KEY` 读取凭据。任何 API Key、数据库、
Chroma 索引、下载原文和本地缓存都被 `.gitignore` 排除。

## 启动旧问答原型

LLM 生成部分仍保留原型的本地 Ollama 与 OpenRouter 路径。检索入口已经切换为
Dense Top50 + BM25 Top50 + RRF(60) + Cohere 重排。Cohere 凭据从
`COHERE_API_KEY` 或本地忽略文件 `backend/cohere.txt` 读取。

```powershell
Set-Location backend
uvicorn FastAPI:app --port 8000 --reload
```

前端为静态页面，可在 `frontend/` 目录启动任意本地 HTTP 服务器。

## 已知限制

- 公开评测是英语报告，不能推断中文企业问答效果。
- 目前的权限与版本实验使用显式元数据上下文，尚未接入真实身份认证。
- 无答案题仍会召回相似文本，尚未实现可靠拒答器。
- 本地离线实验不是并发、吞吐量或生产 SLA 测试。
- 可用 `git switch --detach legacy-vector-baseline` 查看升级前的纯向量版本。
- 详细失败案例见
  [FAILURE_CASES.md](experiments/production_proxy/FAILURE_CASES.md)。
