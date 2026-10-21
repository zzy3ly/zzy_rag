# 异步知识入库与版本生命周期

## 目标

上传接口只负责校验、持久化原文件并创建任务，解析、结构化切分、向量化和索引
构建由后台工作线程完成。新版本完整构建并通过校验之前，查询只读取旧的
`ACTIVE`版本。

普通文件上传按1 MiB缓冲区流式计算SHA-256并落盘，不会把大文件完整读入应用
内存。ZIP批量包仍由当前原型一次性解包，生产部署应增加压缩包大小、文件数和
解压后总量限制。

```text
PENDING -> PARSING -> CHUNKING -> EMBEDDING
        -> VALIDATING -> ACTIVATING -> ACTIVE
              |              |
              +-> RETRYING --+
                      |
                    FAILED
```

每次状态变化都会写入SQLite审计表。服务重启后，未完成任务会恢复为
`PENDING`并重新进入队列。默认线程队列用于单机复现；多实例部署时应将
`IngestionManager`的入队边界替换成Celery/Redis，并将生命周期元数据迁移到
PostgreSQL。业务状态机和向量版本协议不需要变化。

## 幂等与版本

- 逻辑文档由`collection_id + document_key`唯一标识，文件名默认作为
  `document_key`。
- 幂等键由逻辑文档、文件SHA-256和`pipeline_version`共同决定。重复上传返回
  原任务，不重复解析或生成向量。
- 同一逻辑文档内容变化时生成递增版本，旧版本保留为`ARCHIVED`。
- 解析器或切分器升级时修改`pipeline_version`，相同原文件也会重新构建索引。
- 原文件按SHA-256保存；任务记录文件哈希、切分管线版本、块数、失败原因及
  激活时间。

## 结构化父子切分

`StructureChunking.py`优先识别Markdown标题、中文章节、编号条款、自然段和
表格，再构建两级知识单元：

- 子块：连同文档名和章节路径不超过MiniLM的256 token，用于向量和BM25
  检索。只有单个语义块过长时才使用32 token重叠窗口。
- 父块：同一章节内最多约800 token。子块重排命中后按`parent_id`去重并扩展
  为父块，给生成模型提供完整条件、步骤和注意事项。

每个子块保留`source_start/source_end`、`title_path`、`parent_id`、文档版本、
集合和生命周期状态。线上查询固定增加`lifecycle_state=ACTIVE`过滤，因此
`STAGING`块不会提前泄漏到检索结果。

## API

### 提交文件

```http
POST /process_file/
Content-Type: multipart/form-data

collection_id=engineering
document_key=well-control-policy
file=@well-control.txt
```

接口返回HTTP 202、`job_id`、稳定`document_id`和本次`version_id`。
更新时继续传相同的`document_key`，即使上传文件改名也会形成同一逻辑文档的
新版本。

### 查询任务

```http
GET /ingestion_jobs/{job_id}
```

### 失败重试

```http
POST /ingestion_jobs/{job_id}/retry
```

只有`FAILED`任务可以人工重试。自动重试默认最多三次。

### 查看版本与回滚

```http
GET  /documents/{document_id}/versions
POST /documents/{document_id}/rollback
Content-Type: application/json

{"version_id": "..."}
```

回滚只允许选择已经成功激活过的`ACTIVE/ARCHIVED`版本。

## 原子切换边界

Chroma不提供跨记录事务。本项目让查询和版本切换共享进程内锁，在锁内完成旧
版本归档、新版本激活和BM25不可变快照替换；异常时恢复原元数据。因此单进程
查询不会观察到半切换状态。多进程生产部署需要使用支持别名切换的检索服务，
或使用数据库中的活动版本指针作为所有检索分支的强制过滤条件。
