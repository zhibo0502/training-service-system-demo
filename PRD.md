# 产品需求文档（PRD）：离线可训练双轨智能客服系统

> 文档目标：使任意工程师或 AI 代理可依据本文档从零 1:1 复现本系统，并通过三层验收体系确认复现结果的正确性、泛化能力与防退化能力。

---

## 1. 产品概述与运行模式

### 1.1 核心定位
构建一套纯本地优先、可闭环训练的电商智能客服工作台（`training-service-system-demo`），打通 **语料导入/合成 → 清洗去重与类别均衡 → 双轨训练（意图分类 + 向量检索） → 自动化调参 → 线上问答与转人工 → 人工回复反哺训练集** 的完整链路。

### 1.2 运行模式与自动降级
1. **双轨增强模式（`ollama-dual-recall`）**：
   - 同时开启 `USE_OLLAMA=1` 与 `OLLAMA_EMBED=1`，连接本地 Ollama（默认 `http://127.0.0.1:11434`），生成模型默认 `qwen2.5:3b`，向量模型默认 `bge-m3`。
   - 检索引擎采用 `HybridSemanticEmbedder`：语义向量（`OllamaEmbedder`，权重 `0.65`）+ 词法向量（`TrainedTfidfEmbedder`，权重 `0.35`，内部采用平方根加权拼接 + L2 归一化）。
2. **混合生成模式（`ollama-hybrid`）**：
   - 仅开启 `USE_OLLAMA=1`（未开启 `OLLAMA_EMBED=1`）时，由 `qwen2.5:3b` 负责生成，检索使用纯本地 `TrainedTfidfEmbedder`。
3. **纯离线降级模式（`offline-fallback`）**：
   - 未开启 Ollama 或 Ollama 超时/不可用时，通过 `FallbackAwareLlmClient` 与 `HybridSemanticEmbedder` 自动降级为纯本地 `TrainedTfidfEmbedder` + 检索引用模板回复，保证服务零中断。

---

## 2. 技术栈、运行环境与 Monorepo 架构

### 2.1 运行时与启动配置
- **Node.js**：`package.json` 声明 `>= 20.0.0`；因直接使用内置 `node:sqlite` 与免参数原生 TypeScript 剥离运行，**实际推荐 Node.js `>= 22.6.0`（或 `>= 23.6.0`）**。
- **启动命令**：`npm run dev`（默认监听 `http://127.0.0.1:3000`）。
- **环境变量**：
  - `PORT`（默认 `3000`）
  - `USE_OLLAMA`（`1` 启用 Ollama 对话生成）
  - `OLLAMA_EMBED`（`1` 启用 Ollama 向量召回）
  - `OLLAMA_BASE_URL`（默认 `http://127.0.0.1:11434`）
  - `OLLAMA_CHAT_MODEL`（默认 `qwen2.5:3b`）
  - `OLLAMA_EMBED_MODEL`（默认 `bge-m3`）
- **持久化目录布局（`data/`）**：
  - `data/demo.sqlite`：SQLite 结构化存储（文档、数据集版本、样本、训练任务、会话、消息、工单）。
  - `data/vectors.json`：`LocalVectorStore` 向量索引持久化文件。
  - `data/uploads/`：工作台自定义上传语料目录。
  - `data/feedback-qa.jsonl`：人工客服回复反哺生成的标准 QA 语料。

### 2.2 Monorepo 包职责
- `packages/shared`：纯领域模型与 TypeScript 类型定义（`QaCategory`、`QaSample`、`DatasetVersion`、`TrainingJob`、`HandoffTicket`、`ChatMessage` 等）。注：建议同时导出 `export type StandardCategory = QaCategory` 以保证严格 `tsc` 编译通过。
- `packages/collector`：外部语料文件解析与导入（`parseCsvCorpus` 解析 `.csv`，其余扩展名如 `.jsonl`/兼容 JSONL 的 `.txt` 统一走 `parseJsonlCorpus`）及 SQLite 持久化层 `SqliteStore`（基于 `node:sqlite` 的 `DatabaseSync`）。
- `packages/cleaner`：敏感信息脱敏（`sanitizeText`）、归一化去重、类别均衡（`undersample` / `oversample`）、固定随机种子可复现切分（`seededShuffle` + `train/val/test`）、确定性多槽位数据合成引擎（`synthesizer.ts`）。
- `packages/trainer`：意图分类器（`CharNgramNaiveBayesClassifier`）、词法向量化器（`TrainedTfidfEmbedder`，含 20 组同义概念归一化）、`OllamaEmbedder`、`HybridSemanticEmbedder`、`LocalVectorStore`、RAG Hit@3 与 Macro-F1 评测器。
- `packages/agent-core`：客服问答编排（`OfflineAgentService`、`OllamaLlmClient`、`FallbackAwareLlmClient`、Top-8 初筛重排取 Top-3、转人工策略 `handoff-policy.ts`、人工认领回复与反馈记录）。
- `apps/web`：HTTP 服务（`src/server.ts`）与单页三标签页工作台（`public/index.html`）。
- `scripts/`：数据处理、合成与跨集盲测评估脚本（`npm run synth-data`、`npm run build-bayes-data`、`npm run eval-compare`、`npm run build-hybrid-gold`）。

---

## 3. 核心数据契约

### 3.1 意图类别枚举（固定 4 类 `QaCategory`）
- `consultation`（售前咨询）
- `logistics`（物流配送）
- `refund`（退换退款）
- `complaint`（投诉维权 / 破损赔付）

### 3.2 标准语料格式（JSONL）
```json
{
  "question": "string（客户问题）",
  "answer": "string（标准客服回复）",
  "category": "consultation | logistics | refund | complaint（可选，缺失时由规则映射补全）",
  "source": "string（可选，来源标识）"
}
```

### 3.3 SFT 微调导出格式（JSONL）
```json
{
  "instruction": "你是一个严谨、专业的离线电商客服助手，请依据业务规则准确作答。",
  "input": "string（清洗后问题）",
  "output": "string（标准答案）",
  "category": "consultation | logistics | refund | complaint",
  "split": "train | val | test"
}
```

---

## 4. 功能模块与 HTTP 接口契约

### 4.1 全局状态与语料管理
- `GET /`：返回单页工作台 HTML（`apps/web/public/index.html`）。
- `GET /api/overview`：返回运行模式 `mode`（`"ollama-dual-recall" | "ollama-hybrid" | "offline-fallback"`）、`ollama` 状态详情、当前线上模型标签 `activeModel`、已挂载语料列表 `activeSources`、可用预设语料列表 `availablePresets`、文档统计、数据集版本、训练历史、工单与会话。
- `POST /api/corpus/use-preset`：接收 `{ filePath, replaceCurrent }`，校验预设白名单后替换或叠加语料。
- `POST /api/corpus/import`：接收 JSON `{ filename, content }`，写入 `data/uploads/` 并解析去重入库，返回 `{ document, recordCount, deduplicated }`。
- `POST /api/corpus/remove`：接收 `{ filePath }` 或 `{ onlyCustom: true }`，移除指定语料（至少保留一个有效来源）。
- `GET /api/dataset/export-finetune`：导出最新数据集版本为 SFT 微调 JSONL 文件。

### 4.2 Tab 1：客户对话端
- `GET /api/chat/messages?sessionId=...`：拉取指定会话与消息历史（含客户、AI、系统转人工提示、人工回复）。
- `POST /api/chat/send`：发送客户消息 `{ sessionId, customerId, content }`，返回 `{ session, customerMessage, assistantMessage, predictedCategory, ticket?, activeModel }`（置信度 `confidence` 与引用 `citations` 嵌套在 `assistantMessage` 内）。
- `POST /api/chat/feedback`：提交对某条 AI 消息的评价 `{ messageId, feedback }`，其中 `feedback` 枚举值为 `"helpful" | "unhelpful"`。

### 4.3 Tab 2：样本均衡与训练工作台
- **预设语料下拉（`availablePresets`）**，首位为推荐黄金融合集：
  1. `data/hybrid-gold-train.jsonl`（黄金配比融合集 6,107 条 - 推荐）
  2. `packages/cleaner/test/fixtures/common-100-qa.jsonl`（人工均衡基准 100 条）
  3. `data/external/taobao-ecd-qa.jsonl`（淘宝真实电商客服 296 条）
  4. `data/bayes-enhanced-train-8k.jsonl`（复合贝叶斯增强训练集 6,580 条）
  5. `data/bayes-hard-eval-200.jsonl`（复合难点盲测集 158 条）
  6. `data/external/open-real-2k-sample.jsonl`（开源真实混合客服 2,000 条）
  7. `data/corpus-10k.jsonl`（多维槽位合成母本 10,000 条）
- `POST /api/pipeline/run`：按传入 `{ version, seed, balanceStrategy, splitRatio, classifierType, ngramMax, alpha, evalSplit, useSynthAugment }` 执行合并清洗、均衡、切分、训练、向量入库与评测，返回指标（Accuracy、Macro-F1、RAG Hit@3、混淆矩阵）。
- `POST /api/pipeline/autotune`：接收 `{ seed, multiSeed, useSynthAugment, splitRatio }`，执行多参数网格搜索与交叉验证，返回 `{ trials, bestParams, finalResult }`。

### 4.4 Tab 3：人工接管台
- `POST /api/handoff/reply`：人工专员认领工单并发送回复 `{ ticketId, agentId, replyContent }`，状态流转为 `claimed`，消息实时写入原会话。
- `POST /api/handoff/promote-qa`：接收 `{ ticketId }`，将已完成人工回复的工单问答对追加写入 `data/feedback-qa.jsonl`，并自动加入当前激活语料列表用于后续重训。

---

## 5. 核心非功能约束与关键超参数（1:1 复现基线）

1. **冷启动与重启恢复机制**：
   - 首次启动（SQLite 中无数据集版本）时，自动导入 `common-100-qa.jsonl` 并执行 `v1.0.0` 基线训练。
   - 服务重启时，自动查找历史最高 `macroF1` 的训练任务并重建内存分类器与 TF-IDF 词表，保证线上模型不丢失。
2. **黄金配比融合与防泄漏规范（`scripts/build-hybrid-gold.ts`）**：
   - 全量保留真实与人工基准语料（`common-100-qa`、`taobao-ecd-qa`、`open-real-2k`），防止被合成模板淹没口语泛化能力。
   - 对 `bayes-enhanced-train-8k` 每类分层抽样 600 条（`seed=42`），对 `corpus-10k` 每类分层抽样 400 条（`seed=99`），补齐复合意图与多槽位覆盖。
   - 构建时强制剔除与独立盲测集 `bayes-hard-eval-200.jsonl` 归一化相同的任何样本（`skippedBlindLeakage` 透明统计，必须零泄漏）。
3. **线上最优模型防退化保护**：
   - 在 `executePipelineRun` 中比对历史最佳 `Macro-F1`（`prevBestF1`）。仅当首次训练或新模型 `macroF1 >= prevBestF1` 时，才替换线上向量库与线上分类器；否则仅写入实验记录，不降低线上服务质量。
4. **核心算法与检索超参数**：
   - **朴素贝叶斯（`CharNgramNaiveBayesClassifier`）**：IDF 平滑 `Math.log((totalDocs + 1) / (df + 1)) + 1`，`bigramWeight = 2.5`，过滤高频停用单字表 `STOP_CHARS`，默认 `ngramMax = 2`、`alpha = 1.0`。
   - **词法 TF-IDF（`TrainedTfidfEmbedder`）**：包含 20 组 `SEMANTIC_CONCEPT_GROUPS` 同义概念归一化（概念权重 `5.0`，bigram 权重 `2.5`，unigram 权重 `1.0`）。
   - **Web 训练增强（`useSynthAugment = true`）**：`executePipelineRun` 中将 4,000 条合成语料（`baseCount=1200, targetCount=4000, seed=2026`）与 **15 倍重复的真实训练集**拼接训练（`autotune` 交叉验证中为前 800 条合成语料 + **10 倍重复** `foldTrain`）。
   - **AutoTune 搜索空间**：3 种均衡策略 × `ngramMax ∈ [1, 2, 3]` × `alpha ∈ [0.3, 0.5, 1.0, 1.5, 2.0]` + 3 组 `keyword_baseline` = **48 组配置**，在 `[seed, 7, 2026]` × **5 折交叉验证**下寻优。
   - **检索重排与转人工阈值**：Top-8 初筛后按 `score + categoryBoost(0.15) + keywordOverlap(min(0.3, hits*0.08))` 重排取 Top-3；置信度低于 `0.25`、命中转人工/高危词（`冒烟|焦糊|起火|触电|12315|起诉|报警|辱骂`）或连续 2 次 `unhelpful` 触发转人工工单。
5. **敏感信息脱敏**：
   - 清洗阶段必须对身份证号、银行卡号、手机号、邮箱执行正则脱敏，占位符依次为 `[ID_CARD]`、`[BANK_CARD]`、`[PHONE]`、`[EMAIL]`。

---

## 6. 三层验收标准（复现完成定义）

新系统重构完成后，必须依次通过以下三层验证：

1. **第一层：自动化单元与集成测试（`npm test`）**
   - 覆盖脱敏去重、固定 `seed` 切分、合成器均衡、双轨训练与评测、同义归一化与自动降级、`FallbackAwareLlmClient` 降级、Web HTTP 全闭环。
   - 通过标准：全部测试用例 **100% 通过**（0 失败，0 跳过）。
2. **第二层：跨分布泛化与盲测验证（`npm run build-hybrid-gold`）**
   - 同分布 10% 留出集：`Accuracy >= 0.95`、`Macro-F1 >= 0.95`。
   - 复合难点盲测集（`bayes-hard-eval-200.jsonl`，零泄漏）：`Accuracy >= 0.95`、`Macro-F1 >= 0.95`。
   - 全未见外部真实电商语料（`open-real-41k` 抽样）：`Accuracy >= 0.95`，不得因引入合成语料而出现明显下滑。
3. **第三层：模拟人工浏览器端到端验收**
   - 启动服务并访问 `http://127.0.0.1:3000`，全程无 JS 控制台报错、无 HTTP 4xx/5xx 异常：
     - Tab 1：发送问题获得带引用的回复 → 点赞反馈 → 触发转人工生成工单。
     - Tab 2：下拉选中 `hybrid-gold-train.jsonl` → 加载并运行训练评测 → 验证混淆矩阵、训练历史与 SFT 导出可用。
     - Tab 3：认领工单回复 → 点击“收录为 QA 样本” → 切回 Tab 1 验证人工回复实时同步显示。
