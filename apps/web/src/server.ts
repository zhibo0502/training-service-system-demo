import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { dirname, join, resolve } from "node:path";
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
  SqliteStore,
} from "../../../packages/collector/src/index.ts";
import {
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

  // 首次启动时自动导入内置示例语料并生成 v1.0.0 均衡基线
  if (store.listDatasetVersions().length === 0) {
    const imported = importCorpusFile(corpusPath, store, {
      title: "内置中文客服与投诉示例语料",
    });
    const cleaned = runCleaningPipeline(
      imported.records,
      imported.document.id,
      store,
      {
        version: "v1.0.0",
        description: "初始基线：undersample 1:1:1:1 均衡采样",
        seed: 42,
        balanceStrategy: "undersample",
      }
    );
    await runRagIndexJob(cleaned.chunks, embedder, vectorStore, store, {
      datasetVersionId: cleaned.datasetVersion.id,
      seed: 42,
    });
    await runIntentEvalJob(
      cleaned.samples,
      (q) => mapToStandardCategory(undefined, q),
      store,
      {
        datasetVersionId: cleaned.datasetVersion.id,
        seed: 42,
        targetModel: "keyword-baseline-v1",
        hyperparams: { balanceStrategy: "undersample" },
      }
    );
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

      if (req.method === "POST" && pathname === "/api/pipeline/run") {
        const body = (await readJsonBody(req)) as {
          version?: string;
          seed?: number;
          balanceStrategy?: BalanceStrategy;
        };
        const imported = importCorpusFile(corpusPath, store);
        const seed = Number(body.seed ?? 42);
        const strategy: BalanceStrategy = body.balanceStrategy ?? "undersample";
        const version =
          body.version?.trim() ||
          `v1.${store.listDatasetVersions().length}.0`;

        const cleaned = runCleaningPipeline(
          imported.records,
          imported.document.id,
          store,
          {
            version,
            seed,
            balanceStrategy: strategy,
          }
        );

        const ragJob = await runRagIndexJob(
          cleaned.chunks,
          embedder,
          vectorStore,
          store,
          { datasetVersionId: cleaned.datasetVersion.id, seed }
        );

        const evalJob = await runIntentEvalJob(
          cleaned.samples,
          (q) => mapToStandardCategory(undefined, q),
          store,
          {
            datasetVersionId: cleaned.datasetVersion.id,
            seed,
            targetModel: "keyword-baseline-v1",
            hyperparams: { balanceStrategy: strategy },
          }
        );

        return sendJson(res, 200, {
          rawDistribution: cleaned.rawDistribution,
          datasetVersion: cleaned.datasetVersion,
          ragJob,
          evalJob,
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

// 仅当直接运行 server.ts 时监听端口
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
