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
  applyComplianceGuardrail,
  detectSubIntent,
  extractOrderId,
  FallbackAwareLlmClient,
  type LlmClient,
  lookupOrder,
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

    // 1.5 普通投诉提问（不含转人工/安全红线）：应先由 AI 给出安抚与处理方案，不直接转人工
    const complaintAiRes = await agent.handleCustomerMessage({
      sessionId: "sess-1",
      customerId: "cust-100",
      content: "收到商品破损了我要投诉怎么赔付？",
    });
    assert.equal(complaintAiRes.session.status, "ai_active");
    assert.equal(complaintAiRes.assistantMessage.role, "assistant");
    assert.equal(complaintAiRes.ticket, undefined);

    // 2. 触发转人工：用户主动要求转人工
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
    assert.equal(store.listChatMessages("sess-1").length, 7);
  } finally {
    store.close();
  }
});

test("FallbackAwareLlmClient 在主模型不可用时自动降级为离线模板", async () => {
  const brokenPrimary: LlmClient = {
    modelName: "qwen2.5:3b",
    async chat() {
      throw new Error("connect ECONNREFUSED 127.0.0.1:11434");
    },
  };
  const client = new FallbackAwareLlmClient(
    brokenPrimary,
    new TemplateFallbackLlmClient()
  );
  const res = await client.chat([
    { role: "system", content: "标准回答：支持 7 天无理由退货" },
    { role: "user", content: "可以退货吗？" },
  ]);
  assert.match(res.content, /7 天无理由退货/);
  assert.match(res.model, /offline-template-fallback \(fallback from qwen2\.5:3b\)/);
});

test("订单工具、二级意图识别与合规护栏拦截生效", () => {
  assert.equal(extractOrderId("帮我查一下 ord-1001 的物流"), "ORD-1001");
  const order = lookupOrder("ORD-1001");
  assert.equal(order.carrier, "顺丰速运");

  assert.equal(detectSubIntent("我想开发票怎么开", "consultation"), "invoice_request");
  assert.equal(detectSubIntent("快递摔碎了咋赔偿", "complaint"), "damaged_compensation");

  const guard = applyComplianceGuardrail(
    "加我微信 13800138000 私下处理，无条件赔十倍！"
  );
  assert.equal(guard.triggered, true);
  assert.ok(!guard.safeText.includes("13800138000"));
  assert.ok(!guard.safeText.includes("无条件赔十倍"));
});
