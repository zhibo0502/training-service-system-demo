# 培训-客服系统demo (`training-service-system-demo`)

基于 TypeScript 构建的离线客户问答与客服系统最小可行架构（MVP），采用 MIT 开源协议。

> 注：GitHub 仓库名称仅支持 ASCII 字符，因此仓库目录与建议的 GitHub 仓库标识为 `training-service-system-demo`。

## 架构模块

```text
training-service-system-demo/
├── apps/
│   └── web/               # 1. 界面层：客户问答端与管理后台
├── packages/
│   ├── shared/            # 公共类型、数据协议与共享定义
│   ├── collector/         # 2. 数据收集：文档上传与对话日志采集
│   ├── cleaner/           # 3. 数据清洗：正文解析、去重、脱敏与 QA 提取
│   ├── trainer/           # 4. 数据训练：向量索引构建与微调数据集导出
│   └── agent-core/        # 5. 离线客服核心：本地检索、Ollama 推理与人工接管
└── scripts/               # 本地部署与辅助脚本
```

## 开源协议

本项目采用 [MIT License](./LICENSE)。
