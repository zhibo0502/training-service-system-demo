import assert from "node:assert/strict";
import test from "node:test";
import {
  checkSystemInvariants,
  CIRCUIT_TRANSITIONS,
  MODEL_DEPLOY_TRANSITIONS,
  QA_REVIEW_TRANSITIONS,
  SESSION_TRANSITIONS,
  StateTransitionError,
  transitionState,
} from "../src/index.ts";

test("形式化状态机迁移校验：合法迁移通过，非法迁移抛出 StateTransitionError", () => {
  // 1. 会话与接管状态机
  let s = transitionState("Session", SESSION_TRANSITIONS, "ai_active", "TRIGGER_HANDOFF");
  assert.equal(s, "handoff_pending");
  s = transitionState("Session", SESSION_TRANSITIONS, s, "HUMAN_CLAIM");
  assert.equal(s, "human_active");
  s = transitionState("Session", SESSION_TRANSITIONS, s, "RESOLVE_TO_AI");
  assert.equal(s, "ai_active");

  assert.throws(
    () => transitionState("Session", SESSION_TRANSITIONS, "ai_active", "HUMAN_CLAIM"),
    StateTransitionError
  );

  // 2. QA 审核状态机（终态不可逆）
  const approved = transitionState("QaReview", QA_REVIEW_TRANSITIONS, "pending_review", "APPROVE");
  assert.equal(approved, "approved");
  assert.throws(
    () => transitionState("QaReview", QA_REVIEW_TRANSITIONS, approved, "REJECT"),
    StateTransitionError
  );

  // 3. 模型部署状态机
  let m = transitionState("ModelDeploy", MODEL_DEPLOY_TRANSITIONS, "uninitialized", "INIT_BASELINE");
  m = transitionState("ModelDeploy", MODEL_DEPLOY_TRANSITIONS, m, "START_EVAL");
  m = transitionState("ModelDeploy", MODEL_DEPLOY_TRANSITIONS, m, "PROMOTE");
  m = transitionState("ModelDeploy", MODEL_DEPLOY_TRANSITIONS, m, "ROLLBACK");
  assert.equal(m, "rolled_back");
  assert.throws(
    () => transitionState("ModelDeploy", MODEL_DEPLOY_TRANSITIONS, "uninitialized", "ROLLBACK"),
    StateTransitionError
  );

  // 4. 熔断器状态机
  let c = transitionState("Circuit", CIRCUIT_TRANSITIONS, "closed", "FAILURE_THRESHOLD_REACHED");
  c = transitionState("Circuit", CIRCUIT_TRANSITIONS, c, "COOLDOWN_ELAPSED");
  c = transitionState("Circuit", CIRCUIT_TRANSITIONS, c, "PROBE_SUCCEEDED");
  assert.equal(c, "closed");
});

test("系统不变量检查器能捕获状态撕裂与孤立状态", () => {
  const cleanViolations = checkSystemInvariants({
    sessions: [
      {
        id: "s1",
        customerId: "c1",
        channel: "web_widget",
        status: "human_active",
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:00:00Z",
      },
    ],
    tickets: [
      {
        id: "t1",
        sessionId: "s1",
        reason: "user_requested",
        summary: "test",
        status: "claimed",
        createdAt: "2026-01-01T00:00:00Z",
      },
    ],
    activePredictorReady: true,
    vectorCount: 10,
    embedderFitted: true,
    activeModelDatasetId: "ds-1",
    activeVectorDatasetId: "ds-1",
  });
  assert.equal(cleanViolations.length, 0);

  const dirtyViolations = checkSystemInvariants({
    sessions: [
      {
        id: "s2",
        customerId: "c2",
        channel: "web_widget",
        status: "human_active",
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:00:00Z",
      },
    ],
    tickets: [],
    activePredictorReady: false,
    vectorCount: 5,
    embedderFitted: false,
    activeModelDatasetId: "ds-1",
    activeVectorDatasetId: "ds-2",
  });
  assert.equal(dirtyViolations.length, 4);
});
