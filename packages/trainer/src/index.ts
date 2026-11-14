import { randomUUID } from "node:crypto";
import type { SqliteStore } from "@ts-qa/collector";
import type {
  CleanChunk,
  QaCategory,
  QaSample,
  TrainingJob,
} from "@ts-qa/shared";
import type { Embedder } from "./embedder.ts";
import { evaluateIntentClassifier } from "./evaluator.ts";
import type { LocalVectorStore } from "./vector-store.ts";

export * from "./embedder.ts";
export * from "./vector-store.ts";
export * from "./evaluator.ts";
export * from "./naive-bayes-classifier.ts";

/**
 * 轨道 A：执行 RAG 知识块向量化入库，并保存 TrainingJob 记录
 */
export async function runRagIndexJob(
  chunks: readonly CleanChunk[],
  embedder: Embedder,
  vectorStore: LocalVectorStore,
  store: SqliteStore,
  options?: { datasetVersionId?: string; seed?: number }
): Promise<TrainingJob> {
  const startedAt = new Date().toISOString();
  const vectors = [];

  for (const chunk of chunks) {
    const vector = await embedder.embed(chunk.content);
    vectors.push({
      id: chunk.id,
      documentId: chunk.documentId,
      content: chunk.content,
      category: chunk.metadata.sourceLabel,
      vector,
    });
  }

  vectorStore.upsert(vectors);

  const job: TrainingJob = {
    id: randomUUID(),
    jobType: "rag_index",
    datasetVersionId: options?.datasetVersionId,
    seed: options?.seed ?? 42,
    targetModel: embedder.modelName,
    hyperparams: {
      chunkCount: chunks.length,
    },
    sampleCount: chunks.length,
    status: "succeeded",
    createdAt: startedAt,
    finishedAt: new Date().toISOString(),
  };

  store.saveTrainingJob(job);
  return job;
}

/**
 * 轨道 B：在测试集上运行可复现意图/投诉分类评测，并记录实验指标
 */
export async function runIntentEvalJob(
  samples: readonly QaSample[],
  predictCategory: (question: string) => Promise<QaCategory> | QaCategory,
  store: SqliteStore,
  options: {
    datasetVersionId?: string;
    seed?: number;
    targetModel: string;
    evalSplit?: "val" | "test";
    hyperparams?: Record<string, string | number | boolean>;
  }
): Promise<TrainingJob> {
  const startedAt = new Date().toISOString();
  const targetSplit = options.evalSplit ?? "test";
  const splitSamples = samples.filter((s) => s.split === targetSplit);
  const evalTarget = splitSamples.length > 0 ? splitSamples : samples;

  const metrics = await evaluateIntentClassifier(evalTarget, predictCategory);

  const job: TrainingJob = {
    id: randomUUID(),
    jobType: "intent_eval",
    datasetVersionId: options.datasetVersionId,
    seed: options.seed ?? 42,
    targetModel: options.targetModel,
    hyperparams: options.hyperparams ?? {},
    sampleCount: evalTarget.length,
    status: "succeeded",
    metrics,
    createdAt: startedAt,
    finishedAt: new Date().toISOString(),
  };

  store.saveTrainingJob(job);
  return job;
}
