import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { importCorpusFile, SqliteStore } from "../../collector/src/index.ts";
import { runCleaningPipeline } from "../src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(__dirname, "./fixtures/sample-corpus.jsonl");

test("数据收集、脱敏、去重、样本均衡与可复现切分闭环测试", () => {
  const store = new SqliteStore(":memory:");

  try {
    // 1. 首次导入公开格式语料（240 条）
    const firstImport = importCorpusFile(fixturePath, store);
    assert.equal(firstImport.deduplicated, false);
    assert.equal(firstImport.records.length, 240);

    // 2. 再次导入同一文件应触发校验和幂等去重
    const secondImport = importCorpusFile(fixturePath, store);
    assert.equal(secondImport.deduplicated, true);

    // 3. 执行清洗与均衡采样（seed = 42, undersample）
    const run1 = runCleaningPipeline(
      firstImport.records,
      firstImport.document.id,
      store,
      {
        version: "v1.0.0",
        seed: 42,
        balanceStrategy: "undersample",
      }
    );

    // 验证清洗前分布（80/60/55/45）
    assert.deepEqual(run1.rawDistribution, {
      consultation: 80,
      logistics: 60,
      refund: 55,
      complaint: 45,
    });

    // 验证均衡后分布（以最小类 45 条为基准，4 类严格 1:1:1:1 共 180 条）
    assert.deepEqual(run1.datasetVersion.classDistribution, {
      consultation: 45,
      logistics: 45,
      refund: 45,
      complaint: 45,
    });
    assert.equal(run1.samples.length, 180);

    // 验证脱敏规则（手机号与邮箱均应被替换）
    const sanitizedChunks = run1.chunks.filter((c) => c.sanitized);
    for (const chunk of sanitizedChunks) {
      assert.equal(chunk.content.includes("13812345678"), false);
      assert.equal(chunk.content.includes("help@example.com"), false);
    }

    // 验证数据库落库与读回一致
    const savedSamples = store.listQaSamples(run1.datasetVersion.id);
    assert.equal(savedSamples.length, 180);

    // 4. 验证可复现性：使用相同 seed 再跑一次，抽取结果顺序与切分归属完全一致
    const store2 = new SqliteStore(":memory:");
    const run2 = runCleaningPipeline(
      firstImport.records,
      firstImport.document.id,
      store2,
      {
        version: "v1.0.0",
        seed: 42,
        balanceStrategy: "undersample",
      }
    );
    assert.deepEqual(
      run1.samples.map((s) => ({ q: s.question, c: s.category, split: s.split })),
      run2.samples.map((s) => ({ q: s.question, c: s.category, split: s.split }))
    );
    store2.close();
  } finally {
    store.close();
  }
});
