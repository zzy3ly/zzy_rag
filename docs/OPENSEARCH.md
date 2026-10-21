# OpenSearch在线检索后端

在线服务已经从Chroma和进程内BM25迁移到OpenSearch 3.9。历史实验代码继续
保留Chroma依赖，仅用于复现此前报告。

## 本地启动

Docker Compose配置关闭了安全插件，只允许用于本机开发：

```powershell
docker compose -f docker-compose.opensearch.yml up -d
Invoke-RestMethod http://localhost:9200/_cluster/health

Set-Location backend
D:\python\python.exe -m pip install -r requirements.txt
D:\python\python.exe -m pip install -r requirements-opensearch.txt
uvicorn FastAPI:app --port 8000 --reload
```

默认配置：

```text
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX_SHARDS=1
OPENSEARCH_INDEX_REPLICAS=0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COHERE_RERANK_MODEL=rerank-multilingual-v3.0
```

服务启动时创建`rag-chunks-v1`索引和`rag-hybrid-rrf-v1`搜索流水线。编码模型
采用懒加载，首次入库或查询时才初始化。

## 索引设计

- `text`：同时参与BM25检索并作为MiniLM编码输入。
- `embedding`：384维、Lucene HNSW、余弦相似度。
- `document_id/version_id/lifecycle_state`：版本范围和生命周期过滤。
- `collection_id`：租户或知识库范围。
- `parent_id/parent_text`：子块召回后的父块展开。
- `title_path/source_start/source_end`：章节和原文溯源。

查询使用OpenSearch原生`hybrid`查询：BM25 `match`和`knn`各形成一路候选，
`score-ranker-processor`以RRF常数60融合，之后再调用Cohere重排。权限范围和
`lifecycle_state=ACTIVE`作为Hybrid Query预过滤条件，不会在召回之后补过滤。

## 迁移旧Chroma索引

OpenSearch启动后执行：

```powershell
Set-Location backend
D:\python\python.exe -X utf8 migrate_chroma_to_opensearch.py `
  --persist-directory .\chromadb `
  --source-collection crisischatbot `
  --target-index rag-chunks-v1
```

脚本按批次重新生成384维向量并使用相同Chunk ID幂等写入。迁移完成并验证后再
启动新的FastAPI服务。原Chroma目录不会自动删除，可用于回退。

## 生产部署边界

仓库中的Compose是单节点开发环境。企业部署至少需要：

- 开启Security Plugin、TLS和证书校验，不使用明文管理员密码；
- 三个集群管理节点，并根据容量配置数据节点、分片与至少一个副本；
- 为应用账户设置最小索引权限，开启审计日志；
- 配置快照仓库、恢复演练、磁盘水位告警和查询延迟监控；
- 在与生产相同的分片数量下重新评测RRF参数和召回指标；
- 整批重建索引时使用稳定Alias完成无停机切换。
