import { existsSync } from "node:fs";
import { resolve } from "node:path";
import {
  checkSystemInvariants,
  CIRCUIT_TRANSITIONS,
  MODEL_DEPLOY_TRANSITIONS,
  QA_REVIEW_TRANSITIONS,
  SESSION_TRANSITIONS,
  type TransitionTable,
} from "../packages/shared/src/index.ts";
import { SqliteStore } from "../packages/collector/src/index.ts";
import { LocalVectorStore } from "../packages/trainer/src/index.ts";

function exploreFsm<S extends string, E extends string>(
  name: string,
  table: TransitionTable<S, E>,
  initial: S
) {
  const states = Object.keys(table) as S[];
  const visited = new Set<S>([initial]);
  const queue: S[] = [initial];
  let edgeCount = 0;
  const deadEnds: S[] = [];

  while (queue.length > 0) {
    const cur = queue.shift()!;
    const edges = Object.entries(table[cur] ?? {}) as Array<[E, S]>;
    if (edges.length === 0) {
      deadEnds.push(cur);
    }
    for (const [, next] of edges) {
      edgeCount += 1;
      if (!visited.has(next)) {
        visited.add(next);
        queue.push(next);
      }
    }
  }

  const unreachable = states.filter((s) => !visited.has(s));
  return {
    name,
    totalStates: states.length,
    reachableStates: visited.size,
    unreachableStates: unreachable,
    transitions: edgeCount,
    terminalStates: deadEnds,
  };
}

function main() {
  const root = resolve(import.meta.dirname, "..");
  const fsmReports = [
    exploreFsm("SessionHandoffMachine", SESSION_TRANSITIONS, "ai_active"),
    exploreFsm("QaReviewMachine", QA_REVIEW_TRANSITIONS, "pending_review"),
    exploreFsm("ModelDeploymentMachine", MODEL_DEPLOY_TRANSITIONS, "uninitialized"),
    exploreFsm("OllamaCircuitBreaker", CIRCUIT_TRANSITIONS, "closed"),
  ];

  let liveViolations: ReturnType<typeof checkSystemInvariants> = [];
  const dbPath = resolve(root, "data/demo.sqlite");
  const vecPath = resolve(root, "data/vectors.json");

  if (existsSync(dbPath)) {
    const store = new SqliteStore(dbPath);
    const vecStore = new LocalVectorStore(vecPath);
    try {
      liveViolations = checkSystemInvariants({
        sessions: store.listChatSessions(),
        tickets: store.listHandoffTickets(),
        activePredictorReady: store.listTrainingJobs().length > 0,
        vectorCount: vecStore.listRecords().length,
        embedderFitted: vecStore.listRecords().length > 0,
      });
    } finally {
      store.close();
    }
  }

  const hasUnreachable = fsmReports.some((r) => r.unreachableStates.length > 0);
  console.log(
    JSON.stringify(
      {
        ok: !hasUnreachable && liveViolations.length === 0,
        fsmReports,
        liveDbChecked: existsSync(dbPath),
        liveInvariantViolations: liveViolations,
      },
      null,
      2
    )
  );
  if (hasUnreachable || liveViolations.length > 0) {
    process.exit(1);
  }
}

main();
