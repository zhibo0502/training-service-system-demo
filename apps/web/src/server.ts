import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type BalanceStrategy,
  type ChatMessage,
  checkSystemInvariants,
} from "../../../packages/shared/src/index.ts";
import {
  FallbackAwareLlmClient,
  OfflineAgentService,
  OllamaLlmClient,
  TemplateFallbackLlmClient,
} from "../../../packages/agent-core/src/index.ts";
import {
  balanceAndSplitSamples,
  mapToStandardCategory,
  runCleaningPipeline,
  sanitizeText,
  synthesizeQaDataset,
} from "../../../packages/cleaner/src/index.ts";
import {
  importCorpusFile,
  type RawCorpusRecord,
  SqliteStore,
} from "../../../packages/collector/src/index.ts";
import {
  CharNgramNaiveBayesClassifier,
  evaluateIntentClassifier,
  HybridSemanticEmbedder,
  LocalVectorStore,
  OllamaEmbedder,
  runIntentEvalJob,
  runRagIndexJob,
  TrainedTfidfEmbedder,
} from "../../../packages/trainer/src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "../../..");
const defaultCorpusPath = resolve(
  repoRoot,
  "packages/cleaner/test/fixtures/common-100-qa.jsonl"
);
const htmlPath = resolve(__dirname, "../public/index.html");

export interface AppServerOptions {
  dataDir?: string;
  useOllama?: boolean;
  useOllamaEmbed?: boolean;
  ollamaBaseUrl?: string;
  ollamaChatModel?: string;
  ollamaEmbedModel?: string;
  corpusPath?: string;
}

export async function createAppServer(options?: AppServerOptions) {
  const dataDir = options?.dataDir ?? resolve(repoRoot, "data");
  const useOllama = options?.useOllama ?? process.env.USE_OLLAMA === "1";
  const useOllamaEmbed =
    options?.useOllamaEmbed ?? (useOllama && process.env.OLLAMA_EMBED === "1");
  const ollamaBaseUrl =
    options?.ollamaBaseUrl ??
    process.env.OLLAMA_BASE_URL ??
    "http://127.0.0.1:11434";
  const ollamaChatModel =
    options?.ollamaChatModel ?? process.env.OLLAMA_CHAT_MODEL ?? "qwen2.5:3b";
  const ollamaEmbedModel =
    options?.ollamaEmbedModel ?? process.env.OLLAMA_EMBED_MODEL ?? "bge-m3";
  const corpusPath = options?.corpusPath ?? defaultCorpusPath;
  const importedFiles = new Set<string>([corpusPath]);

  const store = new SqliteStore(join(dataDir, "demo.sqlite"));
  const vectorStore = new LocalVectorStore(join(dataDir, "vectors.json"));
  const trainedEmbedder = new TrainedTfidfEmbedder({ ngramMax: 2 });
  const ollamaEmbedder = useOllamaEmbed
    ? new OllamaEmbedder({
        baseUrl: ollamaBaseUrl,
        modelName: ollamaEmbedModel,
      })
    : undefined;
  const embedder = ollamaEmbedder
    ? new HybridSemanticEmbedder({
        semanticEmbedder: ollamaEmbedder,
        lexicalEmbedder: trainedEmbedder,
        semanticWeight: 0.65,
      })
    : trainedEmbedder;
  const ollamaLlm = useOllama
    ? new OllamaLlmClient({
        baseUrl: ollamaBaseUrl,
        modelName: ollamaChatModel,
      })
    : undefined;
  const fallbackLlm = new TemplateFallbackLlmClient();
  const llmClient = ollamaLlm
    ? new FallbackAwareLlmClient(ollamaLlm, fallbackLlm)
    : fallbackLlm;

  const agentService = new OfflineAgentService({
    store,
    embedder,
    vectorStore,
    llmClient,
    minConfidenceThreshold: 0.25,
  });

  let activeModelLabel = "未加载";
  let activePredictorReady = false;
  let embedderFitted = false;
  let activeModelDatasetId: string | undefined;
  let activeVectorDatasetId: string | undefined;

  // 缓存一份确定性合成语料用于增强 Web 端分类器训练集（val/test 仍保持 100% 纯真实样本）
  const cachedSynthTrainSamples = synthesizeQaDataset({
    baseCount: 1200,
    targetCount: 4000,
    seed: 2026,
  }).allRecords.map((r) => ({
    id: r.id,
    datasetVersionId: "synth-aug-4k",
    question: r.question,
    answer: r.answer,
    category: r.rawCategory,
    split: "train" as const,
  }));

  function buildPredictorFromSamples(
    samples: ReturnType<typeof balanceAndSplitSamples>["samples"],
    classifierType: "naive_bayes" | "keyword_baseline",
    ngramMax: number,
    alpha: number,
    useSynthAugment = true
  ) {
    if (classifierType === "naive_bayes") {
      const nb = new CharNgramNaiveBayesClassifier({
        ngramMin: 1,
        ngramMax,
        alpha,
      });
      const trainSamples = samples.filter((s) => s.split === "train");
      const baseTrain = trainSamples.length > 0 ? trainSamples : samples;
      if (useSynthAugment) {
        const weightedBaseTrain = Array.from({ length: 15 }, () => baseTrain).flat();
        nb.fit([...cachedSynthTrainSamples, ...weightedBaseTrain]);
      } else {
        nb.fit(baseTrain);
      }
      return {
        predictFn: (q: string) => nb.predict(q),
        targetModel: `char-ngram-nb${useSynthAugment ? "+synth" : ""}(n=1..${ngramMax},a=${alpha})`,
      };
    }
    return {
      predictFn: (q: string) => mapToStandardCategory(undefined, q),
      targetModel: "keyword-baseline-v1",
    };
  }

  function getBestHistoricalEvalJob() {
    const evalJobs = store
      .listTrainingJobs()
      .filter((j) => j.jobType === "intent_eval" && j.metrics);
    if (evalJobs.length === 0) return undefined;
    return evalJobs.reduce((best, cur) =>
      (cur.metrics?.macroF1 ?? 0) > (best.metrics?.macroF1 ?? 0) ? cur : best
    );
  }

  async function executePipelineRun(params: {
    version: string;
    seed: number;
    balanceStrategy: BalanceStrategy;
    splitRatio: { train: number; val: number; test: number };
    classifierType: "naive_bayes" | "keyword_baseline";
    ngramMax: number;
    alpha: number;
    evalSplit: "val" | "test";
    useSynthAugment?: boolean;
  }) {
    const allRecords: RawCorpusRecord[] = [];
    let lastDocId = "merged-corpus";

    for (const filePath of importedFiles) {
      const imported = importCorpusFile(filePath, store);
      allRecords.push(...imported.records);
      lastDocId = imported.document.id;
    }

    const cleaned = runCleaningPipeline(allRecords, lastDocId, store, {
      version: params.version,
      description: `${params.balanceStrategy}, split=${params.splitRatio.train}/${params.splitRatio.val}/${params.splitRatio.test}`,
      seed: params.seed,
      balanceStrategy: params.balanceStrategy,
      splitRatio: params.splitRatio,
    });

    const { predictFn, targetModel } = buildPredictorFromSamples(
      cleaned.samples,
      params.classifierType,
      params.ngramMax,
      params.alpha,
      params.useSynthAugment ?? true
    );

    const prevBestJob = getBestHistoricalEvalJob();
    const prevBestF1 = prevBestJob?.metrics?.macroF1 ?? -1;

    // 让检索向量直接从当前训练集学习词表与 IDF（受 ngramMax 与均衡策略影响）
    const activeLexicalEmbedder = new TrainedTfidfEmbedder({
      ngramMax: params.ngramMax,
    });
    activeLexicalEmbedder.fit(
      cleaned.chunks.map((c) => c.content),
      params.ngramMax
    );
    const activeEmbedder = ollamaEmbedder
      ? new HybridSemanticEmbedder({
          semanticEmbedder: ollamaEmbedder,
          lexicalEmbedder: activeLexicalEmbedder,
          semanticWeight: 0.65,
        })
      : activeLexicalEmbedder;

    // 先在临时向量索引或当前索引做评测；为保护线上最佳版本，仅当优于等于历史最佳或首次时才替换线上向量与分类器
    const tempVectorStore = new LocalVectorStore();
    const evalVectorStore =
      prevBestJob === undefined ? vectorStore : tempVectorStore;

    const ragJob = await runRagIndexJob(
      cleaned.chunks,
      activeEmbedder,
      evalVectorStore,
      store,
      { datasetVersionId: cleaned.datasetVersion.id, seed: params.seed }
    );

    const evalJob = await runIntentEvalJob(cleaned.samples, predictFn, store, {
      datasetVersionId: cleaned.datasetVersion.id,
      seed: params.seed,
      targetModel,
      evalSplit: params.evalSplit,
      ragEval: { embedder: activeEmbedder, vectorStore: evalVectorStore, topK: 3 },
      hyperparams: {
        balanceStrategy: params.balanceStrategy,
        evalSplit: params.evalSplit,
        trainRatio: params.splitRatio.train,
        valRatio: params.splitRatio.val,
        testRatio: params.splitRatio.test,
        classifierType: params.classifierType,
        ngramMax: params.ngramMax,
        alpha: params.alpha,
      },
    });

    const newF1 = evalJob.metrics?.macroF1 ?? 0;
    const promotedToLive = newF1 >= prevBestF1;
    if (promotedToLive) {
      trainedEmbedder.fit(
        cleaned.chunks.map((c) => c.content),
        params.ngramMax
      );
      embedderFitted = true;
      if (evalVectorStore !== vectorStore) {
        await runRagIndexJob(cleaned.chunks, embedder, vectorStore, store, {
          datasetVersionId: cleaned.datasetVersion.id,
          seed: params.seed,
        });
      }
      agentService.setIntentPredictor(predictFn);
      activePredictorReady = true;
      activeModelDatasetId = cleaned.datasetVersion.id;
      activeVectorDatasetId = cleaned.datasetVersion.id;
      activeModelLabel = `${targetModel} (${params.version})`;
    }

    return {
      rawDistribution: cleaned.rawDistribution,
      datasetVersion: cleaned.datasetVersion,
      ragJob,
      evalJob,
      promotedToLive,
      activeModel: activeModelLabel,
    };
  }

  // 首次启动时自动导入 100 套常用中文客服基准语料并跑通 v1.0.0 可训练基线
  if (store.listDatasetVersions().length === 0) {
    importCorpusFile(corpusPath, store, {
      title: "常用中文客服与投诉基准语料(100套)",
    });
    await executePipelineRun({
      version: "v1.0.0",
      seed: 42,
      balanceStrategy: "undersample",
      splitRatio: { train: 0.8, val: 0.1, test: 0.1 },
      classifierType: "naive_bayes",
      ngramMax: 2,
      alpha: 1.0,
      evalSplit: "test",
    });
  } else {
    for (const doc of store.listRawDocuments()) {
      importedFiles.add(doc.filePath);
    }
    // 重启时自动从历史评测中挑选 Macro-F1 最高的超参数恢复在线推理分类器
    const bestJob = getBestHistoricalEvalJob();
    const bestDataset = bestJob?.datasetVersionId
      ? store.listDatasetVersions().find((v) => v.id === bestJob.datasetVersionId)
      : store.listDatasetVersions()[0];
    if (bestJob && bestDataset) {
      const savedSamples = store.listQaSamples(bestDataset.id);
      const hp = bestJob.hyperparams ?? {};
      const ngramMax = Number(hp.ngramMax ?? 2);
      const { predictFn, targetModel } = buildPredictorFromSamples(
        savedSamples,
        (hp.classifierType as "naive_bayes" | "keyword_baseline") ?? "naive_bayes",
        ngramMax,
        Number(hp.alpha ?? 1.0)
      );
      const storedVectors = vectorStore.listRecords();
      const fitTexts =
        storedVectors.length > 0
          ? storedVectors.map((v) => v.content)
          : savedSamples
              .filter((s) => s.split === "train")
              .map((s) => `问题：${s.question}\n标准回答：${s.answer}`);
      trainedEmbedder.fit(fitTexts, ngramMax);
      embedderFitted = true;
      agentService.setIntentPredictor(predictFn);
      activePredictorReady = true;
      activeModelDatasetId = bestDataset.id;
      activeVectorDatasetId = bestDataset.id;
      activeModelLabel = `${targetModel} (${bestDataset.version})`;
    }
  }

  interface QaReviewItem {
    id: string;
    ticketId: string;
    question: string;
    answer: string;
    category: string;
    status: "pending_review" | "approved" | "rejected";
    createdAt: string;
    reviewedAt?: string;
  }

  const qaReviewPath = join(dataDir, "qa-review-queue.json");
  function loadQaReviewQueue(): QaReviewItem[] {
    if (!existsSync(qaReviewPath)) return [];
    try {
      return JSON.parse(readFileSync(qaReviewPath, "utf-8")) as QaReviewItem[];
    } catch {
      return [];
    }
  }
  function saveQaReviewQueue(items: QaReviewItem[]): void {
    mkdirSync(dataDir, { recursive: true });
    writeFileSync(qaReviewPath, JSON.stringify(items, null, 2), "utf-8");
  }

  const startedAt = Date.now();
  const chatLatencies: number[] = [];
  const runtimeMetrics = {
    totalRequests: 0,
    chatRequests: 0,
    handoffCount: 0,
    guardrailTriggers: 0,
    helpfulCount: 0,
    unhelpfulCount: 0,
  };
  const auditLogs: Array<{
    id: string;
    timestamp: string;
    action: string;
    detail: string;
  }> = [];
  function recordAudit(action: string, detail: string): void {
    auditLogs.unshift({
      id: randomUUID().slice(0, 8),
      timestamp: new Date().toISOString(),
      action,
      detail,
    });
    if (auditLogs.length > 50) auditLogs.length = 50;
  }

  const server = createServer(async (req, res) => {
    try {
      runtimeMetrics.totalRequests += 1;
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const pathname = url.pathname;

      if (req.method === "GET" && pathname === "/healthz") {
        return sendJson(res, 200, {
          status: "ok",
          uptimeSec: Math.round((Date.now() - startedAt) / 1000),
          pid: process.pid,
        });
      }

      if (req.method === "GET" && pathname === "/readyz") {
        const versions = store.listDatasetVersions();
        const vectorCount = vectorStore.listRecords().length;
        const ready = versions.length > 0 && vectorCount > 0;
        return sendJson(res, ready ? 200 : 503, {
          status: ready ? "ready" : "not_ready",
          datasetVersions: versions.length,
          vectorCount,
          activeModel: activeModelLabel,
        });
      }

      if (req.method === "GET" && pathname === "/api/metrics") {
        const sorted = [...chatLatencies].sort((a, b) => a - b);
        const p95LatencyMs =
          sorted.length > 0
            ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))]
            : 0;
        const avgLatencyMs =
          sorted.length > 0
            ? Math.round(sorted.reduce((a, b) => a + b, 0) / sorted.length)
            : 0;
        const fbTotal = runtimeMetrics.helpfulCount + runtimeMetrics.unhelpfulCount;
        return sendJson(res, 200, {
          uptimeSec: Math.round((Date.now() - startedAt) / 1000),
          totalRequests: runtimeMetrics.totalRequests,
          chatRequests: runtimeMetrics.chatRequests,
          p95LatencyMs,
          avgLatencyMs,
          handoffRate:
            runtimeMetrics.chatRequests > 0
              ? Number((runtimeMetrics.handoffCount / runtimeMetrics.chatRequests).toFixed(4))
              : 0,
          positiveFeedbackRate:
            fbTotal > 0
              ? Number((runtimeMetrics.helpfulCount / fbTotal).toFixed(4))
              : 1,
          guardrailTriggers: runtimeMetrics.guardrailTriggers,
          vectorCount: vectorStore.listRecords().length,
          pendingQaReviewCount: loadQaReviewQueue().filter(
            (x) => x.status === "pending_review"
          ).length,
          auditLogs: auditLogs.slice(0, 20),
        });
      }

      if (req.method === "GET" && pathname === "/") {
        const html = readFileSync(htmlPath, "utf-8");
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(html);
        return;
      }

      if (req.method === "GET" && pathname === "/api/overview") {
        const ollamaHealth = ollamaLlm
          ? await ollamaLlm.checkHealth(1500)
          : undefined;
        return sendJson(res, 200, {
          mode: useOllama
            ? useOllamaEmbed
              ? "ollama-dual-recall"
              : "ollama-hybrid"
            : "offline-fallback",
          ollama: {
            enabled: useOllama,
            embedMode: useOllamaEmbed
              ? `hybrid(${ollamaEmbedModel}+trained-tfidf)`
              : "trained-tfidf",
            chatModel: ollamaChatModel,
            embedModel: ollamaEmbedModel,
            available: ollamaHealth?.available ?? false,
            installedModels: ollamaHealth?.models ?? [],
          },
          activeModel: activeModelLabel,
          activeSources: Array.from(importedFiles).map((filePath) => ({
            filePath,
            name: basename(filePath),
            isDefaultDemo: filePath === defaultCorpusPath,
          })),
          availablePresets: [
            {
              label: "hybrid-gold-train.jsonl (黄金配比融合集 6,107 条 - 推荐)",
              filePath: resolve(repoRoot, "data/hybrid-gold-train.jsonl"),
            },
            {
              label: "common-100-qa.jsonl (人工均衡基准 100 条)",
              filePath: defaultCorpusPath,
            },
            {
              label: "taobao-ecd-qa.jsonl (淘宝真实电商客服 296 条)",
              filePath: resolve(repoRoot, "data/external/taobao-ecd-qa.jsonl"),
            },
            {
              label: "bayes-enhanced-train-8k.jsonl (复合贝叶斯增强训练集 6,580 条)",
              filePath: resolve(repoRoot, "data/bayes-enhanced-train-8k.jsonl"),
            },
            {
              label: "bayes-hard-eval-200.jsonl (复合难点盲测集 158 条)",
              filePath: resolve(repoRoot, "data/bayes-hard-eval-200.jsonl"),
            },
            {
              label: "open-real-2k-sample.jsonl (开源真实混合客服 2,000 条)",
              filePath: resolve(repoRoot, "data/external/open-real-2k-sample.jsonl"),
            },
            {
              label: "corpus-10k.jsonl (多维槽位合成母本 10,000 条)",
              filePath: resolve(repoRoot, "data/corpus-10k.jsonl"),
            },
          ].filter((p) => existsSync(p.filePath)),
          documents: store.listRawDocuments(),
          datasetVersions: store.listDatasetVersions(),
          trainingJobs: store.listTrainingJobs(),
          tickets: store.listHandoffTickets(),
          sessions: store.listChatSessions(),
        });
      }

      if (req.method === "POST" && pathname === "/api/corpus/import") {
        const body = (await readJsonBody(req)) as {
          filename?: string;
          content?: string;
        };
        if (!body.filename || !body.content?.trim()) {
          return sendJson(res, 400, { error: "请提供文件名和语料内容" });
        }
        const uploadDir = join(dataDir, "uploads");
        mkdirSync(uploadDir, { recursive: true });
        const safeName = basename(body.filename).replace(/[^a-zA-Z0-9._-]/g, "_");
        const targetPath = join(uploadDir, `${Date.now()}-${safeName}`);
        writeFileSync(targetPath, body.content, "utf-8");
        importedFiles.add(targetPath);

        const imported = importCorpusFile(targetPath, store, {
          title: body.filename,
        });
        return sendJson(res, 200, {
          document: imported.document,
          recordCount: imported.records.length,
          deduplicated: imported.deduplicated,
        });
      }

      if (req.method === "POST" && pathname === "/api/corpus/remove") {
        const body = (await readJsonBody(req)) as {
          filePath?: string;
          onlyCustom?: boolean;
        };
        if (body.onlyCustom) {
          if (importedFiles.size <= 1 && importedFiles.has(defaultCorpusPath)) {
            return sendJson(res, 400, {
              error: "请先上传您自己的真实语料文件，再移除默认演示集。",
            });
          }
          importedFiles.delete(defaultCorpusPath);
        } else if (body.filePath) {
          if (importedFiles.size <= 1 && importedFiles.has(body.filePath)) {
            return sendJson(res, 400, {
              error: "至少需要保留一份训练语料。",
            });
          }
          importedFiles.delete(body.filePath);
        }
        return sendJson(res, 200, {
          ok: true,
          activeCount: importedFiles.size,
        });
      }

      if (req.method === "POST" && pathname === "/api/corpus/use-preset") {
        const body = (await readJsonBody(req)) as {
          filePath?: string;
          replaceCurrent?: boolean;
        };
        const allowedPresets = new Set([
          resolve(repoRoot, "data/hybrid-gold-train.jsonl"),
          defaultCorpusPath,
          resolve(repoRoot, "data/external/taobao-ecd-qa.jsonl"),
          resolve(repoRoot, "data/bayes-enhanced-train-8k.jsonl"),
          resolve(repoRoot, "data/bayes-hard-eval-200.jsonl"),
          resolve(repoRoot, "data/external/open-real-2k-sample.jsonl"),
          resolve(repoRoot, "data/corpus-10k.jsonl"),
        ]);
        if (!body.filePath || !allowedPresets.has(body.filePath) || !existsSync(body.filePath)) {
          return sendJson(res, 400, { error: "所选预设语料不存在或不在允许列表中" });
        }
        if (body.replaceCurrent) {
          importedFiles.clear();
        }
        importedFiles.add(body.filePath);
        const imported = importCorpusFile(body.filePath, store, {
          title: basename(body.filePath),
        });
        return sendJson(res, 200, {
          ok: true,
          document: imported.document,
          recordCount: imported.records.length,
          activeCount: importedFiles.size,
        });
      }

      if (req.method === "GET" && pathname === "/api/dataset/export-finetune") {
        const versions = store.listDatasetVersions();
        const latest = versions[0];
        if (!latest) {
          return sendJson(res, 404, { error: "暂无可导出的数据集版本，请先运行训练。" });
        }
        const samples = store.listQaSamples(latest.id);
        const lines = samples.map((s) =>
          JSON.stringify({
            instruction:
              "你是一个严谨、专业的离线电商客服助手，请依据业务规则准确作答。",
            input: s.question,
            output: s.answer,
            category: s.category,
            split: s.split,
          })
        );
        res.writeHead(200, {
          "Content-Type": "application/x-ndjson; charset=utf-8",
          "Content-Disposition": `attachment; filename="finetune-${latest.version}.jsonl"`,
        });
        res.end(lines.join("\n"));
        return;
      }

      if (req.method === "POST" && pathname === "/api/pipeline/run") {
        const body = (await readJsonBody(req)) as {
          version?: string;
          seed?: number;
          balanceStrategy?: BalanceStrategy;
          splitRatio?: { train: number; val: number; test: number };
          classifierType?: "naive_bayes" | "keyword_baseline";
          ngramMax?: number;
          alpha?: number;
          evalSplit?: "val" | "test";
          useSynthAugment?: boolean;
        };

        const trainR = Number(body.splitRatio?.train ?? 0.8);
        const valR = Number(body.splitRatio?.val ?? 0.1);
        const testR = Number(body.splitRatio?.test ?? 0.1);
        const sum = trainR + valR + testR;

        const result = await executePipelineRun({
          version:
            body.version?.trim() ||
            `v1.${store.listDatasetVersions().length}.0`,
          seed: Number(body.seed ?? 42),
          balanceStrategy: body.balanceStrategy ?? "undersample",
          splitRatio:
            sum > 0
              ? {
                  train: Number((trainR / sum).toFixed(2)),
                  val: Number((valR / sum).toFixed(2)),
                  test: Number((testR / sum).toFixed(2)),
                }
              : { train: 0.8, val: 0.1, test: 0.1 },
          classifierType: body.classifierType ?? "naive_bayes",
          ngramMax: Math.min(3, Math.max(1, Number(body.ngramMax ?? 2))),
          alpha: Math.max(0.05, Number(body.alpha ?? 1.0)),
          evalSplit: body.evalSplit ?? "test",
          useSynthAugment: body.useSynthAugment ?? true,
        });

        return sendJson(res, 200, result);
      }

      if (req.method === "POST" && pathname === "/api/pipeline/autotune") {
        const body = (await readJsonBody(req)) as {
          seed?: number;
          multiSeed?: boolean;
          useSynthAugment?: boolean;
          splitRatio?: { train: number; val: number; test: number };
        };
        const seed = Number(body.seed ?? 42);
        const useMultiSeed = body.multiSeed ?? true;
        const useSynthAugment = body.useSynthAugment ?? true;
        const evalSeeds = useMultiSeed
          ? Array.from(new Set([seed, 7, 2026]))
          : [seed];
        const trainR = Number(body.splitRatio?.train ?? 0.8);
        const valR = Number(body.splitRatio?.val ?? 0.1);
        const testR = Number(body.splitRatio?.test ?? 0.1);
        const sum = trainR + valR + testR;
        const splitRatio =
          sum > 0
            ? {
                train: Number((trainR / sum).toFixed(2)),
                val: Number((valR / sum).toFixed(2)),
                test: Number((testR / sum).toFixed(2)),
              }
            : { train: 0.8, val: 0.1, test: 0.1 };

        // 汇总全部已导入语料并做去重脱敏
        const seen = new Set<string>();
        const candidates = [];
        for (const filePath of importedFiles) {
          const imported = importCorpusFile(filePath, store);
          for (const rec of imported.records) {
            const q = sanitizeText(rec.question);
            const a = sanitizeText(rec.answer);
            if (q.cleanedText.length < 2 || a.cleanedText.length < 2) continue;
            const key = q.cleanedText.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            candidates.push({
              question: q.cleanedText,
              answer: a.cleanedText,
              category: mapToStandardCategory(rec.rawCategory, q.cleanedText),
              sanitized: q.sanitized || a.sanitized,
              intentTag: rec.intentTag,
            });
          }
        }

        const strategies: BalanceStrategy[] = ["undersample", "oversample", "stratified"];
        const ngramList = [1, 2, 3];
        const alphaList = [0.3, 0.5, 1.0, 1.5, 2.0];

        type CandidateConfig = {
          classifierType: "naive_bayes" | "keyword_baseline";
          balanceStrategy: BalanceStrategy;
          ngramMax: number;
          alpha: number;
        };

        const configs: CandidateConfig[] = [];
        for (const balanceStrategy of strategies) {
          configs.push({
            classifierType: "keyword_baseline",
            balanceStrategy,
            ngramMax: 2,
            alpha: 1.0,
          });
          for (const ngramMax of ngramList) {
            for (const alpha of alphaList) {
              configs.push({
                classifierType: "naive_bayes",
                balanceStrategy,
                ngramMax,
                alpha,
              });
            }
          }
        }

        let bestScore = -1;
        let bestParams: CandidateConfig & {
          valMacroF1: number;
          valAccuracy: number;
          seedsEvaluated: number;
        } = {
          classifierType: "naive_bayes",
          balanceStrategy: "undersample",
          ngramMax: 2,
          alpha: 1.0,
          valMacroF1: 0,
          valAccuracy: 0,
          seedsEvaluated: evalSeeds.length,
        };

        const kFolds = 5;
        // 预先为每个 seed × strategy 生成防泄露非测试集，避免内层循环重复切分
        const devCache = new Map<string, ReturnType<typeof balanceAndSplitSamples>["samples"]>();
        for (const s of evalSeeds) {
          for (const balanceStrategy of strategies) {
            const splitOut = balanceAndSplitSamples(candidates, {
              version: "autotune-tmp",
              seed: s,
              balanceStrategy,
              splitRatio,
            });
            devCache.set(
              `${s}-${balanceStrategy}`,
              splitOut.samples.filter((item) => item.split !== "test")
            );
          }
        }

        // 在多种子 × 5 折交叉验证下搜索最稳健超参数，严格锁死 Test 集
        for (const cfg of configs) {
          let totalF1 = 0;
          let totalAcc = 0;
          let evalCount = 0;

          for (const s of evalSeeds) {
            const devSamples = devCache.get(`${s}-${cfg.balanceStrategy}`) ?? [];
            for (let fold = 0; fold < kFolds; fold += 1) {
              const foldVal = devSamples.filter((_, idx) => idx % kFolds === fold);
              const foldTrain = devSamples.filter((_, idx) => idx % kFolds !== fold);
              if (foldVal.length === 0 || foldTrain.length === 0) continue;

              let predictFn: (q: string) => ReturnType<typeof mapToStandardCategory>;
              if (cfg.classifierType === "naive_bayes") {
                const nb = new CharNgramNaiveBayesClassifier({
                  ngramMin: 1,
                  ngramMax: cfg.ngramMax,
                  alpha: cfg.alpha,
                });
                if (useSynthAugment) {
                  const weightedFoldTrain = Array.from(
                    { length: 10 },
                    () => foldTrain
                  ).flat();
                  nb.fit([
                    ...cachedSynthTrainSamples.slice(0, 800),
                    ...weightedFoldTrain,
                  ]);
                } else {
                  nb.fit(foldTrain);
                }
                predictFn = (q) => nb.predict(q);
              } else {
                predictFn = (q) => mapToStandardCategory(undefined, q);
              }

              const metrics = await evaluateIntentClassifier(foldVal, predictFn);
              totalF1 += metrics.macroF1;
              totalAcc += metrics.accuracy;
              evalCount += 1;
            }
          }

          const avgF1 = evalCount > 0 ? Number((totalF1 / evalCount).toFixed(4)) : 0;
          const avgAcc = evalCount > 0 ? Number((totalAcc / evalCount).toFixed(4)) : 0;

          const isBetter =
            avgF1 > bestScore ||
            (avgF1 === bestScore && avgAcc > bestParams.valAccuracy) ||
            (avgF1 === bestScore &&
              avgAcc === bestParams.valAccuracy &&
              cfg.alpha > bestParams.alpha);

          if (isBetter) {
            bestScore = avgF1;
            bestParams = {
              ...cfg,
              valMacroF1: avgF1,
              valAccuracy: avgAcc,
              seedsEvaluated: evalSeeds.length,
            };
          }
        }

        // 用多轮寻优选出的最优配置在主 Seed 的 Test 集做最终盲测并落库
        const finalVersion = `v1.${store.listDatasetVersions().length}.0-auto`;
        const finalResult = await executePipelineRun({
          version: finalVersion,
          seed,
          balanceStrategy: bestParams.balanceStrategy,
          splitRatio,
          classifierType: bestParams.classifierType,
          ngramMax: bestParams.ngramMax,
          alpha: bestParams.alpha,
          evalSplit: "test",
          useSynthAugment,
        });

        return sendJson(res, 200, {
          trials: configs.length,
          bestParams,
          finalResult,
        });
      }

      if (req.method === "GET" && pathname === "/api/chat/messages") {
        const sessionId = url.searchParams.get("sessionId") ?? "demo-session-1";
        return sendJson(res, 200, {
          session: store.getChatSession(sessionId) ?? null,
          messages: store.listChatMessages(sessionId),
        });
      }

      if (req.method === "POST" && pathname === "/api/chat/send") {
        const body = (await readJsonBody(req)) as {
          sessionId?: string;
          customerId?: string;
          content?: string;
        };
        if (!body.content?.trim()) {
          return sendJson(res, 400, { error: "提问内容不能为空" });
        }
        const t0 = Date.now();
        const result = await agentService.handleCustomerMessage({
          sessionId: body.sessionId ?? "demo-session-1",
          customerId: body.customerId ?? "demo-customer",
          content: body.content.trim(),
        });
        const elapsed = Date.now() - t0;
        runtimeMetrics.chatRequests += 1;
        chatLatencies.push(elapsed);
        if (chatLatencies.length > 200) chatLatencies.shift();
        if (result.ticket) runtimeMetrics.handoffCount += 1;
        if (result.guardrailTriggered) runtimeMetrics.guardrailTriggers += 1;
        return sendJson(res, 200, {
          ...result,
          latencyMs: elapsed,
          activeModel: activeModelLabel,
        });
      }

      if (req.method === "POST" && pathname === "/api/chat/feedback") {
        const body = (await readJsonBody(req)) as {
          messageId?: string;
          feedback?: ChatMessage["feedback"];
        };
        if (!body.messageId || !body.feedback) {
          return sendJson(res, 400, { error: "缺少 messageId 或 feedback" });
        }
        agentService.submitFeedback(body.messageId, body.feedback);
        if (body.feedback === "helpful") runtimeMetrics.helpfulCount += 1;
        if (body.feedback === "unhelpful") runtimeMetrics.unhelpfulCount += 1;
        return sendJson(res, 200, { ok: true });
      }

      if (req.method === "POST" && pathname === "/api/handoff/reply") {
        const body = (await readJsonBody(req)) as {
          ticketId?: string;
          agentId?: string;
          replyContent?: string;
        };
        if (!body.ticketId || !body.replyContent?.trim()) {
          return sendJson(res, 400, { error: "缺少工单号或回复内容" });
        }
        const result = agentService.claimAndReplyAsHuman({
          ticketId: body.ticketId,
          agentId: body.agentId ?? "human-agent-01",
          replyContent: body.replyContent.trim(),
        });
        recordAudit("handoff_reply", `工单 ${body.ticketId} 已由 ${body.agentId ?? "human-agent-01"} 回复`);
        return sendJson(res, 200, result);
      }

      if (req.method === "POST" && pathname === "/api/handoff/promote-qa") {
        const body = (await readJsonBody(req)) as { ticketId?: string };
        if (!body.ticketId) {
          return sendJson(res, 400, { error: "缺少 ticketId" });
        }
        const ticket = store.getHandoffTicket(body.ticketId);
        if (!ticket) {
          return sendJson(res, 404, { error: "工单不存在" });
        }
        const messages = store.listChatMessages(ticket.sessionId);
        const lastCustomer = [...messages]
          .reverse()
          .find((m) => m.role === "customer");
        const lastHuman = [...messages]
          .reverse()
          .find((m) => m.role === "human_agent");
        if (!lastCustomer || !lastHuman) {
          return sendJson(res, 400, { error: "缺少完整的客户提问或人工回复" });
        }

        const category = mapToStandardCategory(undefined, lastCustomer.content);
        const queue = loadQaReviewQueue();
        const item: QaReviewItem = {
          id: randomUUID().slice(0, 8),
          ticketId: body.ticketId,
          question: lastCustomer.content,
          answer: lastHuman.content,
          category,
          status: "pending_review",
          createdAt: new Date().toISOString(),
        };
        queue.unshift(item);
        saveQaReviewQueue(queue);
        recordAudit("qa_submitted_for_review", `工单 ${body.ticketId} 已提交至 QA 审核队列 (${item.id})`);

        return sendJson(res, 200, {
          ok: true,
          status: "pending_review",
          item,
          question: item.question,
          answer: item.answer,
          category,
        });
      }

      if (req.method === "GET" && pathname === "/api/qa-review/list") {
        return sendJson(res, 200, { items: loadQaReviewQueue() });
      }

      if (req.method === "POST" && pathname === "/api/qa-review/decide") {
        const body = (await readJsonBody(req)) as {
          id?: string;
          decision?: "approve" | "reject";
          question?: string;
          answer?: string;
          category?: string;
        };
        if (!body.id || (body.decision !== "approve" && body.decision !== "reject")) {
          return sendJson(res, 400, { error: "缺少 id 或无效的 decision" });
        }
        const queue = loadQaReviewQueue();
        const idx = queue.findIndex((x) => x.id === body.id);
        if (idx === -1) {
          return sendJson(res, 404, { error: "审核记录不存在" });
        }
        const item = queue[idx];
        item.status = body.decision === "approve" ? "approved" : "rejected";
        item.reviewedAt = new Date().toISOString();
        if (body.question?.trim()) item.question = body.question.trim();
        if (body.answer?.trim()) item.answer = body.answer.trim();
        if (body.category?.trim()) item.category = body.category.trim();
        saveQaReviewQueue(queue);

        if (body.decision === "approve") {
          const feedbackPath = join(dataDir, "feedback-qa.jsonl");
          mkdirSync(dataDir, { recursive: true });
          if (!existsSync(feedbackPath)) {
            writeFileSync(feedbackPath, "", "utf-8");
          }
          appendFileSync(
            feedbackPath,
            `${JSON.stringify({
              question: item.question,
              answer: item.answer,
              category: item.category,
            })}\n`,
            "utf-8"
          );
          importedFiles.add(feedbackPath);
          importCorpusFile(feedbackPath, store, {
            title: "人工客服反哺 QA 语料（已审核）",
          });
          recordAudit("qa_approved", `QA 样本 ${item.id} (${item.category}) 审核通过并入库`);
        } else {
          recordAudit("qa_rejected", `QA 样本 ${item.id} 已驳回`);
        }
        return sendJson(res, 200, { ok: true, item });
      }

      if (req.method === "POST" && pathname === "/api/models/rollback") {
        const body = (await readJsonBody(req)) as { jobId?: string };
        if (!body.jobId) {
          return sendJson(res, 400, { error: "缺少 jobId" });
        }
        const job = store.listTrainingJobs().find((j) => j.id === body.jobId);
        if (!job) {
          return sendJson(res, 404, { error: "目标训练任务不存在" });
        }
        const samples = store.listQaSamples(job.datasetVersionId);
        if (samples.length === 0) {
          return sendJson(res, 400, { error: "该版本关联样本为空，无法回滚" });
        }
        const clsType =
          job.targetModel.includes("keyword") ? "keyword_baseline" : "naive_bayes";
        const ngramMax = Number(job.hyperparams.ngramMax ?? 2);
        const { predictFn, targetModel } = buildPredictorFromSamples(
          samples,
          clsType,
          ngramMax,
          Number(job.hyperparams.alpha ?? 1.0),
          true
        );
        const trainChunks = samples
          .filter((s) => s.split === "train")
          .map((s, i) => ({
            id: `rb-${job.datasetVersionId}-${i}`,
            documentId: s.sourceDocId || job.datasetVersionId,
            chunkIndex: i,
            content: `问题：${s.question}\n标准回答：${s.answer}`,
            sanitized: false,
            metadata: { sourceLabel: s.category },
          }));
        if (trainChunks.length > 0) {
          trainedEmbedder.fit(
            trainChunks.map((c) => c.content),
            ngramMax
          );
          embedderFitted = true;
          await runRagIndexJob(trainChunks, embedder, vectorStore, store, {
            datasetVersionId: job.datasetVersionId,
            seed: job.seed,
          });
        }
        agentService.setIntentPredictor(predictFn);
        activePredictorReady = true;
        activeModelDatasetId = job.datasetVersionId;
        activeVectorDatasetId = job.datasetVersionId;
        const versionObj = store
          .listDatasetVersions()
          .find((v) => v.id === job.datasetVersionId);
        activeModelLabel = `${targetModel} (${versionObj?.version ?? job.id.slice(0, 8)}-rollback)`;
        recordAudit("model_rollback", `线上模型与向量库已同步回滚至 ${activeModelLabel}`);
        return sendJson(res, 200, {
          ok: true,
          activeModel: activeModelLabel,
          jobId: job.id,
        });
      }

      if (req.method === "GET" && pathname === "/api/invariants") {
        const violations = checkSystemInvariants({
          sessions: store.listChatSessions(),
          tickets: store.listHandoffTickets(),
          activePredictorReady,
          vectorCount: vectorStore.listRecords().length,
          embedderFitted,
          activeVectorDatasetId,
          activeModelDatasetId,
        });
        return sendJson(res, 200, {
          ok: violations.length === 0,
          checkedRules: 5,
          violations,
        });
      }

      sendJson(res, 404, { error: "Not Found" });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      sendJson(res, 500, { error: message });
    }
  });

  server.on("close", () => {
    store.close();
  });

  return server;
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf-8").trim();
  return raw ? JSON.parse(raw) : {};
}

const isMain =
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  const port = Number(process.env.PORT ?? 3000);
  const host = process.env.HOST ?? "127.0.0.1";
  const server = await createAppServer();
  server.listen(port, host, () => {
    console.log(
      `培训-客服系统demo 已启动: http://${host}:${port} (离线模式)`
    );
  });
  const shutdown = (signal: string) => {
    console.log(`收到 ${signal}，正在优雅关闭服务...`);
    server.close(() => {
      process.exit(0);
    });
    setTimeout(() => process.exit(1), 5000).unref();
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}
