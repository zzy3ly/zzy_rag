# zzy_rag

一个可复现的油气领域 RAG 检索实验项目。在线服务使用OpenSearch统一承载
BM25、MiniLM向量、元数据过滤和原生RRF混合检索，并包含异步知识入库、
结构化父子切分、内容哈希幂等、失败重试及文档版本切换。历史实验仍保留
Chroma基线，保证原有结果可以复现。

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
  -> 持久化任务队列与状态机
  -> 结构解析与父子分块
  -> OpenSearch / MiniLM 暂存版本
  -> 完整性校验与原子激活
  -> Dense + BM25 候选召回
  -> RRF 融合
  -> Cohere Rerank
  -> LLM 生成与引用
```

- `backend/`：原 FastAPI RAG 原型及新增检索模块。
- `backend/TokenChunking.py`：按嵌入模型 token 上限切分并保留源字符位置。
- `backend/StructureChunking.py`：章节/条款优先的父子切分，小块召回、父块扩展。
- `backend/Ingestion.py`：持久化异步任务、幂等、重试、版本激活、审计与回滚。
- `backend/OpenSearchBackend.py`：索引映射、批量写入、原生BM25+k-NN RRF和版本操作。
- `backend/MiniLMEncoder.py`：本地384维MiniLM编码及与分块共用的Tokenizer。
- `backend/HybridRetrieval.py`：历史实验使用的本地BM25/RRF实现，不再用于在线链路。
- `docs/INGESTION_LIFECYCLE.md`：入库状态机、接口和一致性边界。
- `docs/OPENSEARCH.md`：本地启动、配置、旧索引迁移和生产部署边界。
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

重排实验优先从环境变量 `COHERE_API_KEY` 读取凭据。任何 API Key、本地数据库、
历史Chroma索引、OpenSearch数据卷、下载原文和缓存都不会提交到Git。

## 启动旧问答原型

LLM生成部分仍保留原型的本地Ollama与OpenRouter路径。在线检索由OpenSearch
执行BM25 + k-NN + RRF(60)，随后使用Cohere重排。先按
[OpenSearch部署说明](docs/OPENSEARCH.md)启动服务并配置环境变量。

```powershell
Set-Location backend
D:\python\python.exe -m pip install -r requirements.txt
D:\python\python.exe -m pip install -r requirements-opensearch.txt
uvicorn FastAPI:app --port 8000 --reload
```

`POST /process_file/`现在返回HTTP 202和`job_id`。通过
`GET /ingestion_jobs/{job_id}`查询处理状态；版本历史、失败重试和回滚接口见
[异步入库与版本生命周期说明](docs/INGESTION_LIFECYCLE.md)。本地默认使用
SQLite WAL和一个后台工作线程，运行数据保存在被Git忽略的
`backend/ingestion_data/`。

前端为静态页面，可在 `frontend/` 目录启动任意本地 HTTP 服务器。

## 已知限制

- 公开评测是英语报告，不能推断中文企业问答效果。
- 目前的权限与版本实验使用显式元数据上下文，尚未接入真实身份认证。
- 默认任务执行器面向单机演示；多实例生产部署需要Redis/Celery、PostgreSQL和
  支持跨进程原子版本路由的检索后端。
- 无答案题仍会召回相似文本，尚未实现可靠拒答器。
- 本地离线实验不是并发、吞吐量或生产 SLA 测试。
- 可用 `git switch --detach legacy-vector-baseline` 查看升级前的纯向量版本。
- 详细失败案例见
  [FAILURE_CASES.md](experiments/production_proxy/FAILURE_CASES.md)。
