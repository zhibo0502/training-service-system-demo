# 培训-客服系统demo (`training-service-system-demo`)

基于 TypeScript 构建的离线中文客户问答与客服系统最小可行架构（MVP），采用 MIT 开源协议。

> 注：GitHub 仓库名称仅支持 ASCII 字符，因此仓库目录与建议的 GitHub 仓库标识为 `training-service-system-demo`。

## 快速启动与演示

环境要求：**Node.js >= 22**（零第三方依赖，直接使用原生 `node:sqlite`、`node:http` 与 TypeScript 直跑能力）。

1. **运行全量自动化测试**
   ```bash
   npm test
   ```

2. **启动离线演示 Web 界面**
   ```bash
   npm run dev
   ```
   启动后访问：`http://127.0.0.1:3000`
   - 默认以零依赖离线兜底模式启动，自动导入内置示例语料并生成 `v1.0.0` 均衡基线（`seed = 42`）。
   - 如本地已启动 Ollama（`qwen2.5:3b` 与 `nomic-embed-text`），可切换为真实本地大模型推理：
     ```bash
     USE_OLLAMA=1 npm run dev
     ```

## 架构模块

```text
training-service-system-demo/
├── apps/
│   └── web/               # 1. 界面层：客户问答端、样本均衡看板、人工客服接管台
├── packages/
│   ├── shared/            # 公共类型、数据协议与实验指标定义
│   ├── collector/         # 2. 数据收集：JSONL/CSV 语料导入与 SQLite 持久化
│   ├── cleaner/           # 3. 数据清洗：脱敏、去重、4 类别映射、均衡采样与切分
│   ├── trainer/           # 4. 数据训练：向量索引入库、Macro-F1 评测与 JSONL 导出
│   └── agent-core/        # 5. 离线客服核心：检索问答、Ollama 推理与人工接管状态机
├── docs/
│   └── design.md          # 架构设计文档
└── scripts/               # 本地部署与辅助脚本
```

## 开源协议

本项目采用 [MIT License](./LICENSE)。
