import type { ChatMessage, HandoffReason } from "@ts-qa/shared";

export interface HandoffDecision {
  shouldHandoff: boolean;
  reason?: HandoffReason;
}

/**
 * 评估当前对话是否应触发转人工客服
 */
export function evaluateHandoffNeed(params: {
  customerText: string;
  topScore: number;
  minConfidenceThreshold?: number;
  recentMessages?: readonly ChatMessage[];
}): HandoffDecision {
  const threshold = params.minConfidenceThreshold ?? 0.35;
  const text = params.customerText.trim();

  if (/转人工|人工客服|找人工|找主管|我要投诉|立即投诉/.test(text)) {
    return {
      shouldHandoff: true,
      reason: "user_requested",
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
