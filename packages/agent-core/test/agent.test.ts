import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { runCleaningPipeline } from "../../cleaner/src/index.ts";
import { importCorpusFile, SqliteStore } from "../../collector/src/index.ts";
import {
  LocalNgramEmbedder,
  LocalVectorStore,
  runRagIndexJob,
} from "../../trainer/src/index.ts";
import {
  OfflineAgentService,
  TemplateFallbackLlmClient,
} from "../src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(
  __dirname,
  "../../cleaner/test/fixtures/sample-corpus.jsonl"
);

test("离线客服闭环：检索问答、转人工工单生成与人工客服接管回复", async () => {
  const store = new SqliteStore(":memory:");

  try {
    const imported = importCorpusFile(fixturePath, store);
    const cleaned = runCleaningPipeline(
      imported.records,
      imported.document.id,
      store,
      { version: "v1.0.0", seed: 42, balanceStrategy: "undersample" }
    );

    const embedder = new LocalNgramEmbedder();
    const vectorStore = new LocalVectorStore();
    await runRagIndexJob(cleaned.chunks, embedder, vectorStore, store);

    const agent = new OfflineAgentService({
      store,
      embedder,
      vectorStore,
      llmClient: new TemplateFallbackLlmClient(),
      minConfidenceThreshold: 0.25,
    });

    // 1. 正常提问：应由 AI 回答并附带引用
    const normalRes = await agent.handleCustomerMessage({
      sessionId: "sess-1",
      customerId: "cust-100",
      content: "请问优惠券过期了还能补发吗？",
    });

    assert.equal(normalRes.session.status, "ai_active");
    assert.equal(normalRes.assistantMessage.role, "assistant");
    assert.ok((normalRes.assistantMessage.citations?.length ?? 0) > 0);
    assert.equal(normalRes.ticket, undefined);

    // 2. 触发转人工：用户主动要求投诉/转人工
    const handoffRes = await agent.handleCustomerMessage({
      sessionId: "sess-1",
      customerId: "cust-100",
      content: "我要投诉你们的服务，请立即转人工！",
    });

    assert.equal(handoffRes.session.status, "handoff_pending");
    assert.equal(handoffRes.session.handoffReason, "user_requested");
    assert.ok(handoffRes.ticket);
    assert.equal(handoffRes.ticket.status, "open");

    // 3. 人工客服在后台认领工单并回复
    const humanRes = agent.claimAndReplyAsHuman({
      ticketId: handoffRes.ticket.id,
      agentId: "agent-007",
      replyContent: "您好，我是人工客服 007，已为您加急核实处理。",
    });

    assert.equal(humanRes.ticket.status, "claimed");
    assert.equal(humanRes.session.status, "human_active");
    assert.equal(humanRes.message.role, "human_agent");
    assert.equal(store.listChatMessages("sess-1").length, 5);
  } finally {
    store.close();
  }
});
