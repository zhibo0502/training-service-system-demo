import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { BalanceStrategy, ChatMessage } from "@ts-qa/shared";
import {
  OfflineAgentService,
  OllamaLlmClient,
  TemplateFallbackLlmClient,
} from "../../../packages/agent-core/src/index.ts";
import {
  mapToStandardCategory,
  runCleaningPipeline,
} from "../../../packages/cleaner/src/index.ts";
import {
  importCorpusFile,
  type RawCorpusRecord,
  SqliteStore,
} from "../../../packages/collector/src/index.ts";
import {
  CharNgramNaiveBayesClassifier,
  LocalNgramEmbedder,
  LocalVectorStore,
  OllamaEmbedder,
  runIntentEvalJob,
  runRagIndexJob,
} from "../../../packages/trainer/src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "../../..");
const defaultCorpusPath = resolve(
  repoRoot,
  "packages/cleaner/test/fixtures/sample-corpus.jsonl"
);
const htmlPath = resolve(__dirname, "../public/index.html");

export interface AppServerOptions {
  dataDir?: string;
  useOllama?: boolean;
  corpusPath?: string;
}

export async function createAppServer(options?: AppServerOptions) {
  const dataDir = options?.dataDir ?? resolve(repoRoot, "data");
  const useOllama = options?.useOllama ?? process.env.USE_OLLAMA === "1";
  const corpusPath = options?.corpusPath ?? defaultCorpusPath;
  const importedFiles = new Set<string>([corpusPath]);

  const store = new SqliteStore(join(dataDir, "demo.sqlite"));
  const vectorStore = new LocalVectorStore(join(dataDir, "vectors.json"));
  const embedder = useOllama
    ? new OllamaEmbedder()
    : new LocalNgramEmbedder();
  const llmClient = useOllama
    ? new OllamaLlmClient()
    : new TemplateFallbackLlmClient();

  const agentService = new OfflineAgentService({
    store,
    embedder,
    vectorStore,
    llmClient,
    minConfidenceThreshold: 0.25,
  });

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

    const ragJob = await runRagIndexJob(
      cleaned.chunks,
      embedder,
      vectorStore,
      store,
      { datasetVersionId: cleaned.datasetVersion.id, seed: params.seed }
    );

    let predictFn: (q: string) => ReturnType<typeof mapToStandardCategory>;
    let targetModel = "keyword-baseline-v1";

    if (params.classifierType === "naive_bayes") {
      const nb = new CharNgramNaiveBayesClassifier({
        ngramMin: 1,
        ngramMax: params.ngramMax,
        alpha: params.alpha,
      });
      const trainSamples = cleaned.samples.filter((s) => s.split === "train");
      nb.fit(trainSamples.length > 0 ? trainSamples : cleaned.samples);
      predictFn = (q) => nb.predict(q);
      targetModel = `char-ngram-nb(n=1..${params.ngramMax},a=${params.alpha})`;
    } else {
      predictFn = (q) => mapToStandardCategory(undefined, q);
    }

    // 热更新在线客服使用的意图识别器，打通“训练 -> 推理”闭环
    agentService.setIntentPredictor(predictFn);

    const evalJob = await runIntentEvalJob(cleaned.samples, predictFn, store, {
      datasetVersionId: cleaned.datasetVersion.id,
      seed: params.seed,
      targetModel,
      evalSplit: params.evalSplit,
      ragEval: { embedder, vectorStore, topK: 3 },
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

    return {
      rawDistribution: cleaned.rawDistribution,
      datasetVersion: cleaned.datasetVersion,
      ragJob,
      evalJob,
    };
  }

  // 首次启动时自动导入 240 条基准语料并跑通 v1.0.0 可训练基线
  if (store.listDatasetVersions().length === 0) {
    importCorpusFile(corpusPath, store, {
      title: "内置中文客服与投诉基准语料(240条)",
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
        return sendJson(res, 200, {
          mode: useOllama ? "ollama" : "offline-fallback",
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
        return sendJson(res, 200, result);
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
