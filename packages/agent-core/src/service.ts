import { randomUUID } from "node:crypto";
import type { SqliteStore } from "@ts-qa/collector";
import type {
  ChatMessage,
  ChatSession,
  CitationRef,
  HandoffTicket,
  QaCategory,
} from "@ts-qa/shared";
import type { Embedder, LocalVectorStore } from "@ts-qa/trainer";
import { evaluateHandoffNeed } from "./handoff-policy.ts";
import type { LlmClient } from "./llm-client.ts";

export interface AgentReplyResult {
  session: ChatSession;
  customerMessage: ChatMessage;
  assistantMessage: ChatMessage;
  predictedCategory?: QaCategory;
  ticket?: HandoffTicket;
}

/**
 * 离线客服核心编排服务：意图识别 -> 检索 -> 推理/转人工 -> 人工接管
 */
export class OfflineAgentService {
  private readonly store: SqliteStore;
  private readonly embedder: Embedder;
  private readonly vectorStore: LocalVectorStore;
  private readonly llmClient: LlmClient;
  private readonly minConfidenceThreshold: number;
  private intentPredictor?: (text: string) => QaCategory;

  constructor(deps: {
    store: SqliteStore;
    embedder: Embedder;
    vectorStore: LocalVectorStore;
    llmClient: LlmClient;
    minConfidenceThreshold?: number;
    intentPredictor?: (text: string) => QaCategory;
  }) {
    this.store = deps.store;
    this.embedder = deps.embedder;
    this.vectorStore = deps.vectorStore;
    this.llmClient = deps.llmClient;
    this.minConfidenceThreshold = deps.minConfidenceThreshold ?? 0.35;
    this.intentPredictor = deps.intentPredictor;
  }

  setIntentPredictor(predictor: (text: string) => QaCategory): void {
    this.intentPredictor = predictor;
  }

  async handleCustomerMessage(params: {
    sessionId: string;
    customerId: string;
    content: string;
    channel?: ChatSession["channel"];
  }): Promise<AgentReplyResult> {
    const now = new Date().toISOString();
    let session = this.store.getChatSession(params.sessionId);

    if (!session) {
      session = {
        id: params.sessionId,
        customerId: params.customerId,
        channel: params.channel ?? "web_widget",
        status: "ai_active",
        createdAt: now,
        updatedAt: now,
      };
      this.store.upsertChatSession(session);
    }

    const customerMessage: ChatMessage = {
      id: randomUUID(),
      sessionId: session.id,
      role: "customer",
      content: params.content,
      feedback: null,
      createdAt: now,
    };
    this.store.saveChatMessage(customerMessage);

    // 若会话已由人工接管，只记录客户消息，不再由 AI 抢答
    if (session.status === "human_active" || session.status === "handoff_pending") {
      const waitMsg: ChatMessage = {
        id: randomUUID(),
        sessionId: session.id,
        role: "system",
        content: "您的消息已发送给人工客服，请稍候。",
        feedback: null,
        createdAt: new Date().toISOString(),
      };
      this.store.saveChatMessage(waitMsg);
      return { session, customerMessage, assistantMessage: waitMsg };
    }

    const queryVec = await this.embedder.embed(params.content);
    const rawHits = this.vectorStore.search(queryVec, 8);
    const recentMessages = this.store.listChatMessages(session.id);
    const predictedCategory = this.intentPredictor
      ? this.intentPredictor(params.content)
      : undefined;

    // 结合识别到的意图类别对检索结果重排，减少跨类别误命中（如物流时效误匹配退款到账天数）
    const hits = rawHits
      .map((h) => {
        const categoryBoost =
          predictedCategory && h.record.category === predictedCategory
            ? 0.15
            : 0;
        const keywordBoost = computeKeywordOverlap(
          params.content,
          h.record.content
        );
        return {
          ...h,
          rerankScore: h.score + categoryBoost + keywordBoost,
        };
      })
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .slice(0, 3);

    const topScore = hits[0]?.score ?? 0;

    const decision = evaluateHandoffNeed({
      customerText: params.content,
      topScore,
      predictedCategory,
      minConfidenceThreshold: this.minConfidenceThreshold,
      recentMessages,
    });

    if (decision.shouldHandoff && decision.reason) {
      const ticketTime = new Date().toISOString();
      const ticket: HandoffTicket = {
        id: randomUUID(),
        sessionId: session.id,
        reason: decision.reason,
        summary: `客户提问：${params.content.slice(0, 80)}`,
        status: "open",
        createdAt: ticketTime,
      };
      this.store.upsertHandoffTicket(ticket);

      session = {
        ...session,
        status: "handoff_pending",
        handoffReason: decision.reason,
        updatedAt: ticketTime,
      };
      this.store.upsertChatSession(session);

      const guidance = buildHandoffGuidance(params.content, ticket.id, decision.reason);
      const handoffMessage: ChatMessage = {
        id: randomUUID(),
        sessionId: session.id,
        role: "system",
        content: guidance,
        confidence: Number(topScore.toFixed(4)),
        feedback: null,
        createdAt: ticketTime,
      };
      this.store.saveChatMessage(handoffMessage);

      return {
        session,
        customerMessage,
        assistantMessage: handoffMessage,
        predictedCategory,
        ticket,
      };
    }

    const citations: CitationRef[] = hits.map((h) => ({
      chunkId: h.record.id,
      documentId: h.record.documentId,
      score: Number(h.score.toFixed(4)),
      snippet: h.record.content,
    }));

    const contextBlock = citations
      .map((c, idx) => `[参考${idx + 1}] ${c.snippet}`)
      .join("\n\n");

    const historyMessages = recentMessages
      .filter((m) => m.role === "customer" || m.role === "assistant")
      .slice(-4)
      .map((m) => ({
        role: (m.role === "customer" ? "user" : "assistant") as "user" | "assistant",
        content: m.content,
      }));

    const llmReply = await this.llmClient.chat([
      {
        role: "system",
        content: `你是专业、耐心的离线中文客服助手（当前识别意图：${predictedCategory ?? "general"}）。请严格依据以下参考知识回答客户问题，不要编造未提及的规则：\n\n${contextBlock}`,
      },
      ...historyMessages,
      {
        role: "user",
        content: params.content,
      },
    ]);

    const replyTime = new Date().toISOString();
    const assistantMessage: ChatMessage = {
      id: randomUUID(),
      sessionId: session.id,
      role: "assistant",
      content: llmReply.content,
      confidence: Number(topScore.toFixed(4)),
      citations,
      feedback: null,
      createdAt: replyTime,
    };
    this.store.saveChatMessage(assistantMessage);

    session = {
      ...session,
      updatedAt: replyTime,
    };
    this.store.upsertChatSession(session);

    return {
      session,
      customerMessage,
      assistantMessage,
      predictedCategory,
    };
  }

  claimAndReplyAsHuman(params: {
    ticketId: string;
    agentId: string;
    replyContent: string;
  }): { ticket: HandoffTicket; session: ChatSession; message: ChatMessage } {
    const ticket = this.store.getHandoffTicket(params.ticketId);
    if (!ticket) {
      throw new Error(`工单不存在: ${params.ticketId}`);
    }

    const session = this.store.getChatSession(ticket.sessionId);
    if (!session) {
      throw new Error(`关联会话不存在: ${ticket.sessionId}`);
    }

    const now = new Date().toISOString();
    const updatedTicket: HandoffTicket = {
      ...ticket,
      status: "claimed",
      assignedAgentId: params.agentId,
    };
    this.store.upsertHandoffTicket(updatedTicket);

    const updatedSession: ChatSession = {
      ...session,
      status: "human_active",
      assignedAgentId: params.agentId,
      updatedAt: now,
    };
    this.store.upsertChatSession(updatedSession);

    const message: ChatMessage = {
      id: randomUUID(),
      sessionId: session.id,
      role: "human_agent",
      content: params.replyContent,
      feedback: null,
      createdAt: now,
    };
    this.store.saveChatMessage(message);

    return {
      ticket: updatedTicket,
      session: updatedSession,
      message,
    };
  }

  submitFeedback(
    messageId: string,
    feedback: NonNullable<ChatMessage["feedback"]>
  ): void {
    this.store.updateMessageFeedback(messageId, feedback);
  }
}

function computeKeywordOverlap(query: string, docText: string): number {
  const tokens = query
    .replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, "")
    .split("");
  if (tokens.length === 0) return 0;
  let hits = 0;
  for (let i = 0; i < tokens.length - 1; i += 1) {
    const bigram = tokens[i] + tokens[i + 1];
    if (docText.includes(bigram)) {
      hits += 1;
    }
  }
  return Math.min(0.3, hits * 0.08);
}

function buildHandoffGuidance(
  customerText: string,
  ticketId: string,
  reason: string
): string {
  if (/质量|破损|坏了|瑕疵|投诉/.test(customerText)) {
    return `非常抱歉给您带来不佳体验，我们高度重视您反馈的质量/投诉问题。\n已为您优先创建加急人工工单（${ticketId}，原因：${reason}）。\n建议您先准备好订单号与商品问题照片，人工专员接手后将第一时间为您核实赔付或退换处理。`;
  }
  return `已为您转接人工客服（工单号：${ticketId}，原因：${reason}），请稍候，人工专员正在接入。`;
}
