import type { ChatSession, HandoffTicket } from "./models/conversation.ts";

export class StateTransitionError extends Error {
  readonly machine: string;
  readonly from: string;
  readonly event: string;

  constructor(machine: string, from: string, event: string, reason?: string) {
    super(
      `[${machine}] 非法状态迁移: ${from} --(${event})--> ?${reason ? ` (${reason})` : ""}`
    );
    this.name = "StateTransitionError";
    this.machine = machine;
    this.from = from;
    this.event = event;
  }
}

export type TransitionTable<S extends string, E extends string> = Record<
  S,
  Partial<Record<E, S>>
>;

export function transitionState<S extends string, E extends string>(
  machine: string,
  table: TransitionTable<S, E>,
  current: S,
  event: E
): S {
  const next = table[current]?.[event];
  if (!next) {
    throw new StateTransitionError(machine, current, event);
  }
  return next;
}

/**
 * 1. 会话与人工接管状态机
 */
export type SessionState = ChatSession["status"];
export type SessionEvent =
  | "TRIGGER_HANDOFF"
  | "HUMAN_CLAIM"
  | "RESOLVE_TO_AI";

export const SESSION_TRANSITIONS: TransitionTable<SessionState, SessionEvent> =
  {
    ai_active: {
      TRIGGER_HANDOFF: "handoff_pending",
    },
    handoff_pending: {
      HUMAN_CLAIM: "human_active",
      RESOLVE_TO_AI: "ai_active",
    },
    human_active: {
      RESOLVE_TO_AI: "ai_active",
    },
  };

/**
 * 2. 人工反哺 QA 审核状态机
 */
export type QaReviewState = "pending_review" | "approved" | "rejected";
export type QaReviewEvent = "APPROVE" | "REJECT";

export const QA_REVIEW_TRANSITIONS: TransitionTable<
  QaReviewState,
  QaReviewEvent
> = {
  pending_review: {
    APPROVE: "approved",
    REJECT: "rejected",
  },
  approved: {},
  rejected: {},
};

/**
 * 3. 线上模型部署与回滚状态机
 */
export type ModelDeployState =
  | "uninitialized"
  | "baseline_ready"
  | "candidate_evaluating"
  | "promoted_live"
  | "kept_best"
  | "rolled_back";

export type ModelDeployEvent =
  | "INIT_BASELINE"
  | "START_EVAL"
  | "PROMOTE"
  | "REJECT_BELOW_BEST"
  | "ROLLBACK";

export const MODEL_DEPLOY_TRANSITIONS: TransitionTable<
  ModelDeployState,
  ModelDeployEvent
> = {
  uninitialized: {
    INIT_BASELINE: "baseline_ready",
  },
  baseline_ready: {
    START_EVAL: "candidate_evaluating",
    ROLLBACK: "rolled_back",
  },
  candidate_evaluating: {
    PROMOTE: "promoted_live",
    REJECT_BELOW_BEST: "kept_best",
  },
  promoted_live: {
    START_EVAL: "candidate_evaluating",
    ROLLBACK: "rolled_back",
  },
  kept_best: {
    START_EVAL: "candidate_evaluating",
    ROLLBACK: "rolled_back",
  },
  rolled_back: {
    START_EVAL: "candidate_evaluating",
    ROLLBACK: "rolled_back",
  },
};

/**
 * 4. Ollama 降级熔断器状态机
 */
export type CircuitState = "closed" | "open" | "half_open";
export type CircuitEvent =
  | "FAILURE_THRESHOLD_REACHED"
  | "COOLDOWN_ELAPSED"
  | "PROBE_SUCCEEDED"
  | "PROBE_FAILED";

export const CIRCUIT_TRANSITIONS: TransitionTable<CircuitState, CircuitEvent> =
  {
    closed: {
      FAILURE_THRESHOLD_REACHED: "open",
    },
    open: {
      COOLDOWN_ELAPSED: "half_open",
    },
    half_open: {
      PROBE_SUCCEEDED: "closed",
      PROBE_FAILED: "open",
    },
  };

export interface InvariantViolation {
  rule: string;
  detail: string;
}

/**
 * 形式化系统不变量检查器：
 * 1) 会话处于 handoff_pending / human_active 时必须存在对应工单
 * 2) 工单为 claimed 时，会话必须处于 human_active
 * 3) 线上向量词表空间必须与线上模型关联版本一致
 */
export function checkSystemInvariants(snapshot: {
  sessions: readonly ChatSession[];
  tickets: readonly HandoffTicket[];
  activePredictorReady: boolean;
  vectorCount: number;
  embedderFitted: boolean;
  activeVectorDatasetId?: string;
  activeModelDatasetId?: string;
}): InvariantViolation[] {
  const violations: InvariantViolation[] = [];
  const ticketsBySession = new Map<string, HandoffTicket[]>();

  for (const t of snapshot.tickets) {
    const list = ticketsBySession.get(t.sessionId) ?? [];
    list.push(t);
    ticketsBySession.set(t.sessionId, list);
  }

  for (const s of snapshot.sessions) {
    const linked = ticketsBySession.get(s.id) ?? [];
    if (
      (s.status === "handoff_pending" || s.status === "human_active") &&
      linked.length === 0
    ) {
      violations.push({
        rule: "INV_HANDOFF_TICKET_REQUIRED",
        detail: `会话 ${s.id} 状态为 ${s.status}，但缺少关联工单`,
      });
    }
    if (
      s.status === "ai_active" &&
      linked.some((t) => t.status === "open")
    ) {
      violations.push({
        rule: "INV_NO_OPEN_TICKET_IN_AI_ACTIVE",
        detail: `会话 ${s.id} 处于 ai_active，但仍存在未处理 open 工单`,
      });
    }
  }

  const sessionMap = new Map(snapshot.sessions.map((s) => [s.id, s]));
  for (const t of snapshot.tickets) {
    const sess = sessionMap.get(t.sessionId);
    if (t.status === "claimed" && sess && sess.status !== "human_active") {
      violations.push({
        rule: "INV_CLAIMED_TICKET_REQUIRES_HUMAN_ACTIVE",
        detail: `工单 ${t.id} 已认领(claimed)，但会话 ${sess.id} 状态为 ${sess.status}`,
      });
    }
  }

  if (snapshot.vectorCount > 0 && !snapshot.embedderFitted) {
    violations.push({
      rule: "INV_VECTOR_REQUIRES_FITTED_EMBEDDER",
      detail: "向量库非空但词法向量器处于 unfitted 状态，检索空间错位",
    });
  }

  if (
    snapshot.activeVectorDatasetId &&
    snapshot.activeModelDatasetId &&
    snapshot.activeVectorDatasetId !== snapshot.activeModelDatasetId
  ) {
    violations.push({
      rule: "INV_MODEL_VECTOR_VERSION_SYNC",
      detail: `线上分类器版本(${snapshot.activeModelDatasetId}) 与向量库版本(${snapshot.activeVectorDatasetId}) 不一致`,
    });
  }

  if (snapshot.vectorCount > 0 && !snapshot.activePredictorReady) {
    violations.push({
      rule: "INV_PREDICTOR_READY",
      detail: "向量库已就绪但线上意图分类器未初始化",
    });
  }

  return violations;
}
