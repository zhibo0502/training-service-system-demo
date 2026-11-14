import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { BalanceStrategy, ChatMessage } from "@ts-qa/shared";
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

  function buildPredictorFromSamples(
    samples: ReturnType<typeof balanceAndSplitSamples>["samples"],
    classifierType: "naive_bayes" | "keyword_baseline",
    ngramMax: number,
    alpha: number
  ) {
    if (classifierType === "naive_bayes") {
      const nb = new CharNgramNaiveBayesClassifier({
        ngramMin: 1,
        ngramMax,
        alpha,
      });
      const trainSamples = samples.filter((s) => s.split === "train");
      nb.fit(trainSamples.length > 0 ? trainSamples : samples);
      return {
        predictFn: (q: string) => nb.predict(q),
        targetModel: `char-ngram-nb(n=1..${ngramMax},a=${alpha})`,
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
      params.alpha
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
      if (evalVectorStore !== vectorStore) {
        await runRagIndexJob(cleaned.chunks, embedder, vectorStore, store, {
          datasetVersionId: cleaned.datasetVersion.id,
          seed: params.seed,
        });
      }
      agentService.setIntentPredictor(predictFn);
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
      agentService.setIntentPredictor(predictFn);
      activeModelLabel = `${targetModel} (${bestDataset.version})`;
    }
  }

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const pathname = url.pathname;

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
        });

        return sendJson(res, 200, result);
      }

      if (req.method === "POST" && pathname === "/api/pipeline/autotune") {
        const body = (await readJsonBody(req)) as {
          seed?: number;
          multiSeed?: boolean;
          splitRatio?: { train: number; val: number; test: number };
        };
        const seed = Number(body.seed ?? 42);
        const useMultiSeed = body.multiSeed ?? true;
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
                nb.fit(foldTrain);
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
        const result = await agentService.handleCustomerMessage({
          sessionId: body.sessionId ?? "demo-session-1",
          customerId: body.customerId ?? "demo-customer",
          content: body.content.trim(),
        });
        return sendJson(res, 200, {
          ...result,
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

        const feedbackPath = join(dataDir, "feedback-qa.jsonl");
        mkdirSync(dataDir, { recursive: true });
        if (!existsSync(feedbackPath)) {
          writeFileSync(feedbackPath, "", "utf-8");
        }
        const category = mapToStandardCategory(undefined, lastCustomer.content);
        appendFileSync(
          feedbackPath,
          `${JSON.stringify({
            question: lastCustomer.content,
            answer: lastHuman.content,
            category,
          })}\n`,
          "utf-8"
        );
        importedFiles.add(feedbackPath);
        importCorpusFile(feedbackPath, store, {
          title: "人工客服反哺 QA 语料",
        });

        return sendJson(res, 200, {
          ok: true,
          question: lastCustomer.content,
          answer: lastHuman.content,
          category,
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
  const server = await createAppServer();
  server.listen(port, "127.0.0.1", () => {
    console.log(
      `培训-客服系统demo 已启动: http://127.0.0.1:${port} (离线模式)`
    );
  });
}
