import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { mapToStandardCategory, runCleaningPipeline } from "../../cleaner/src/index.ts";
import { importCorpusFile, SqliteStore } from "../../collector/src/index.ts";
import {
  type Embedder,
  exportFinetuneJsonl,
  HybridSemanticEmbedder,
  LocalNgramEmbedder,
  LocalVectorStore,
  runIntentEvalJob,
  runRagIndexJob,
  TrainedTfidfEmbedder,
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

test("TrainedTfidfEmbedder 同义概念归一化可召回复杂口语同义表达且 HybridSemanticEmbedder 支持自动降级", async () => {
  const docs = [
    { id: "doc-damaged", content: "问题：收到玻璃杯破损怎么申请赔付？\n标准回答：请拍照联系售后理赔。" },
    { id: "doc-fee", content: "问题：退货运费由谁承担？\n标准回答：质量问题商家承担，个人原因运费险理赔。" },
    { id: "doc-stuck", content: "问题：快递停在中转站超过48小时不更新怎么催促？\n标准回答：我们将发起催件工单。" },
    { id: "doc-invoice", content: "问题：下单后如何申请开具电子发票？\n标准回答：订单页填写抬头税号即可开票。" },
    { id: "doc-missing", content: "问题：收到的商品少发漏发配件怎么补发？\n标准回答：核实重量后第一时间补发。" },
  ];

  const embedder = new TrainedTfidfEmbedder({ ngramMax: 2 });
  embedder.fit(docs.map((d) => d.content));

  const store = new LocalVectorStore();
  const records = await Promise.all(
    docs.map(async (d) => ({
      id: d.id,
      documentId: "syn-test",
      content: d.content,
      vector: await embedder.embed(d.content),
    }))
  );
  store.replaceAll(records);

  const queries: Array<[string, string]> = [
    ["东西摔碎了咋赔偿？", "doc-damaged"],
    ["寄回去的邮费谁掏？", "doc-fee"],
    ["包裹卡在半路两天不动了", "doc-stuck"],
    ["买完想开个报销电子票", "doc-invoice"],
    ["箱子里缺了一个零件咋办", "doc-missing"],
  ];

  for (const [q, expectedId] of queries) {
    const hits = store.search(await embedder.embed(q), 3);
    assert.ok(hits.length > 0);
    assert.equal(hits[0].record.id, expectedId, `期望 ${q} Top-1 命中 ${expectedId}，实际为 ${hits[0].record.id}`);
    assert.ok(hits[0].score > 0.25, `期望 ${q} 相似度 > 0.25，实际为 ${hits[0].score}`);
  }

  // HybridSemanticEmbedder 在语义服务异常时无缝降级
  const brokenSemantic: Embedder = {
    modelName: "broken-ollama",
    async embed() {
      throw new Error("ECONNREFUSED");
    },
  };
  const hybrid = new HybridSemanticEmbedder({
    semanticEmbedder: brokenSemantic,
    lexicalEmbedder: embedder,
  });
  const fallbackVec = await hybrid.embed("东西摔碎了咋赔偿？");
  const hits = store.search(fallbackVec, 1);
  assert.equal(hits[0].record.id, "doc-damaged");
});
