# 油气知识库生产场景代理评测 v2

本实验评估检索，不运行回答生成，不代表中石油线上系统。旧实验位于
`../petroleum_retrieval`，原业务 `backend/Embeddings.py` 保留。

## 数据来源与取样

- FORCE NETWORK Group，*Large Oil and Gas industry text dataset from Norwegian,
  UK and Dutch public oil and gas documents*，2024-03-03 v1。
  来源：[Zenodo](https://zenodo.org/records/10775273)，DOI `10.5281/zenodo.10775273`。
- 使用其中 Norway / Norwegian Petroleum Directorate relinquishment reports CSV。
  Zenodo API 记录声明 CC-BY-4.0；原元数据保存在 `raw/zenodo_record.json`。
  下载 MD5：`f6ca32637b9117b2f18b5578c02b55be`。
- 下载的 UK CSV 仅作为后续外部验证资料，本轮未参与知识库或结果。
- 599 份来源报告中，按导出页数 8–30、原文字符数 15,000–65,000 筛选；
  对符合条件的文件名做 SHA256 排序，取前 40 份。不是根据检索分数选文档。
- 40 份报告的 **全部可用导出页** 进入知识库，包括目录、图表 OCR 和噪声；
  共 758 页。8 份报告存在导出页码缺口，不能声称恢复了完整 PDF。
  是否还遗漏尾页、图像内容，未与原 PDF 逐页核验。
- 前 20 份报告各选择 3 个可核验事实，人工式 AI 编写直接问法、改写问法，
  另将前两个事实组合为两部分问题：20 × 7 = 140 道公开资料题。
  60 个基础事实与 20 个组合题并非 140 个独立事实。未声称专家金标。
- 另外有 50 条明确标注 `SIMULATED` 的模拟行政归档记录，分为 10 个资产系列；
  每系列 8 道题，共 80 道。字段值和模板人为设置，不是企业真实制度，
  也不是油气设备操作或安全指令。
- 总计 220 题。按来源报告/资产系列分组，44 题开发、176 题测试；
  测试中公开题 112、模拟题 64。开发集与测试集不共享来源系列。
  所有报告仍在检索库中，划分的是评测问题，符合检索评测需要。

## 问题与边界

公开题覆盖对象名称、井号、专业术语、业务改写、表格行和两处证据。
模拟题覆盖当前版本、历史时间、近似设备编号、表格、两份材料、
无权限、库内无答案和缺站点需要澄清。

`facts.tsv` 保存人工式 AI 编写的问法和逐字证据。`data/queries.jsonl`
保存源页字符偏移与备选证据组。每个题目要求的多个证据组全部相关；
同组多个位置是可替代证据。这里只自动补充了同一报告内逐字重复的证据，
没有完成跨全文所有语义等价证据的专家池化审阅，因此指标是
**已标注证据覆盖率**，不能当作穷尽相关文档的绝对召回率。

问答本身不进入索引。源报告原本的总结段落仍保留。标题使用来源文件名，
所有方法采用相同标题前缀。不要将这轮英语评测推断为中文工业问答效果。

## 对照方法

1. `page_dense`：整页 OCR + MiniLM，展示长文本截断影响的诊断基线。
2. `legacy1000_dense`：原项目 RecursiveCharacterTextSplitter，
   1,000 字符、0 overlap，加同样标题后嵌入。
3. `token_dense`：按模型 tokenizer 切分，标题、正文、特殊 token 合计
   ≤256；重叠 32 token；记录原页字符偏移。
4. `bm25`：相同 token 分块，k1=1.5、b=0.75。
5. `hybrid`：相同 token 分块；dense 50 + BM25 50，RRF 常数 60，
   去重后最多 100 候选。单路对照最多 100 候选。

模型为原项目使用的 all-MiniLM-L6-v2，384 维、归一化。ONNX CPU 执行，
四线程；缓存 `.npy` 向量与原文分块。这轮用精确向量搜索控制 ANN
近似误差，而非运行原 FastAPI 端点，不能将耗时称为服务性能。
原业务方法并未被替换，本轮是可复现的离线对照实现。

所有正式分支使用同一用户上下文做角色、有效期和站点过滤；上下文是测试
输入，未实现自然语言时间解析或接入真实身份认证。
同一资产编号在多个站点出现且输入缺站点时，元数据规则要求澄清。
`ungated_hybrid` 是明确不应用这些约束的对照，不是推荐的服务实现。
过滤规则只在本离线实验中接入，没有升级现有 HTTP 认证。

## 指标与重排

- Evidence Recall@K：前 K 块完整覆盖的已标注证据组数 / 所需组数。
  同一源页相邻块的覆盖可以合并；命中同页的其他位置不算正确。
- All Evidence@K：该题所有所需组均被覆盖的比例。
- Hit@K：至少一个组被完整覆盖的比例。它不能代表多证据问题已解决。
- @100 是第一阶段候选覆盖，不能拿来冒充最终回答上下文的 @5。
- 无答案、无权限、澄清题不进入普通召回分母。
- 按来源系列进行 5,000 次配对 bootstrap，而非假设改写题相互独立。
  样本较小，区间也不代表整个业务查询分布。

重排子集在查看得分前确定：每个测试公开报告，取一个按 ID 哈希选择的
改写题和一个多证据题，32 题。三种适配分块方法采用同一
Cohere `rerank-english-v3.0`，Top5、阈值 0.01。
这是偏向改写与多证据的挑战子集，不是随机业务流量样本，不与 112 题混算。
只发送公开资料，凭证本地读取，不写入输出。API 缓存保留。

权限暴露与错误版本统计针对返回文本，而非回答。无答案题仍可能检索到
看似相关的材料：本实验没有训练可靠拒答器，不能宣称已经解决幻觉。

## 复现

在项目根目录使用已有 Python 环境：

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

重排需有效的 `COHERE_API_KEY`，或项目已有本地凭证文件；默认优先使用环境
变量。既有依赖版本见上一轮实验的 requirements，本轮额外直接使用
tokenizers / onnxruntime（随现有 Chroma 安装）和 langchain-text-splitters。
`results/environment.json` 记录本次实际版本及可复现文件哈希。

## 构建记录和局限

- 构建协议在检索前写入 `PROTOCOL.md`。
- 首次完整性测试发现合成 other-site 记录沿用了原站点元数据。
  在查看评测得分前修正了 site/title，重新冻结并重新运行索引。
- 未根据得分删题、替换失败题、调整融合权重或改写金标。
- 场景比例、合成规则和问法来自开发者设计，没有企业日志或领域专家验证。
  不能仅凭这轮实验声称“已成为大型央企生产级系统”。
- 生产部署还需要业务专家审题、中文数据验证、真实鉴权、增量索引、
  并发与延迟测试。本实验仅支持检索职责的技术论证。
