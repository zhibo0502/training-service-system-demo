import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { mapToStandardCategory, runCleaningPipeline } from "../../cleaner/src/index.ts";
import { importCorpusFile, SqliteStore } from "../../collector/src/index.ts";
import {
  exportFinetuneJsonl,
  LocalNgramEmbedder,
  LocalVectorStore,
  runIntentEvalJob,
  runRagIndexJob,
} from "../src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(
  __dirname,
  "../../cleaner/test/fixtures/sample-corpus.jsonl"
);

test("双轨训练与评测：RAG 向量入库、检索、Macro-F1 评测与 JSONL 导出", async () => {
  const tempDir = mkdtempSync(join(tmpdir(), "ts-qa-trainer-"));
  const store = new SqliteStore(":memory:");

  try {
    const imported = importCorpusFile(fixturePath, store);
    const cleaned = runCleaningPipeline(
      imported.records,
      imported.document.id,
      store,
      { version: "v1.0.0", seed: 42, balanceStrategy: "undersample" }
    );

    // 轨道 A：向量入库与持久化检索
    const embedder = new LocalNgramEmbedder();
    const vectorPath = join(tempDir, "vectors.json");
    const vectorStore = new LocalVectorStore(vectorPath);

    const ragJob = await runRagIndexJob(
      cleaned.chunks,
      embedder,
      vectorStore,
      store,
      { datasetVersionId: cleaned.datasetVersion.id, seed: 42 }
    );

    assert.equal(ragJob.status, "succeeded");
    assert.equal(vectorStore.size(), cleaned.chunks.length);

    const queryVec = await embedder.embed("优惠券过期了还能补发吗？");
    const hits = vectorStore.search(queryVec, 1);
    assert.equal(hits.length, 1);
    assert.ok(hits[0].score > 0.3);

    // 轨道 B：可复现分类评测与指令微调 JSONL 导出
    const evalJob = await runIntentEvalJob(
      cleaned.samples,
      (question) => mapToStandardCategory(undefined, question),
      store,
      {
        datasetVersionId: cleaned.datasetVersion.id,
        seed: 42,
        targetModel: "keyword-baseline-v1",
      }
    );

    assert.equal(evalJob.status, "succeeded");
    assert.ok(evalJob.metrics);
    assert.ok(evalJob.metrics.evaluatedCount > 0);
    assert.ok(evalJob.metrics.macroF1 >= 0 && evalJob.metrics.macroF1 <= 1);

    const jsonlPath = join(tempDir, "train.jsonl");
    const exportedCount = exportFinetuneJsonl(cleaned.samples, jsonlPath);
    assert.equal(exportedCount, cleaned.samples.length);
    const firstLine = JSON.parse(readFileSync(jsonlPath, "utf-8").split("\n")[0]);
    assert.ok(firstLine.instruction && firstLine.input && firstLine.output);

    // 验证训练任务已落库
    assert.equal(store.listTrainingJobs().length, 2);
  } finally {
    store.close();
    rmSync(tempDir, { recursive: true, force: true });
  }
});
