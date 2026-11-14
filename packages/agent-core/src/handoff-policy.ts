import type { ChatMessage, HandoffReason, QaCategory } from "@ts-qa/shared";

export interface HandoffDecision {
  shouldHandoff: boolean;
  reason?: HandoffReason;
}

/**
 * 评估当前对话是否应触发转人工客服
 * 原则：常见投诉先由 AI 给出安抚与标准处理方案，仅在明确要求、高危安全/监管升级、低置信度或连续点踩时转人工
 */
export function evaluateHandoffNeed(params: {
  customerText: string;
  topScore: number;
  predictedCategory?: QaCategory;
  minConfidenceThreshold?: number;
  recentMessages?: readonly ChatMessage[];
}): HandoffDecision {
  const threshold = params.minConfidenceThreshold ?? 0.25;
  const text = params.customerText.trim();

  if (/转人工|人工客服|找人工|找主管|人工专员/.test(text)) {
    return {
      shouldHandoff: true,
      reason: "user_requested",
    };
  }

  // 安全隐患或监管/极端升级场景直接触发人工介入
  if (/冒烟|焦糊|起火|触电|12315|起诉|报警|辱骂/.test(text)) {
    return {
      shouldHandoff: true,
      reason: "policy_blocked",
    };
  }

  const recent = params.recentMessages ?? [];
  const unhelpfulStreak = recent
    .filter((m) => m.role === "assistant")
    .slice(-2)
    .filter((m) => m.feedback === "unhelpful").length;

  if (unhelpfulStreak >= 2) {
    return {
      shouldHandoff: true,
      reason: "repeated_failure",
    };
  }

  if (params.topScore < threshold) {
    return {
      shouldHandoff: true,
      reason: "low_confidence",
    };
  }

  return { shouldHandoff: false };
}
