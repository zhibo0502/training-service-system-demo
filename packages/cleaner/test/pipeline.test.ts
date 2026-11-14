import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { importCorpusFile, SqliteStore } from "../../collector/src/index.ts";
import { balanceAndSplitSamples, runCleaningPipeline } from "../src/index.ts";

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

    // 验证训练集均衡后分布（先切分 8:1:1 后，min(train)=36，训练集 4 类严格 1:1:1:1）
    assert.deepEqual(run1.datasetVersion.classDistribution, {
      consultation: 36,
      logistics: 36,
      refund: 36,
      complaint: 36,
    });
    assert.equal(run1.chunks.length, 144);

    // 验证防泄露：oversample 模式下 train 集与 test 集严格零交集
    const oversampleRun = balanceAndSplitSamples(
      firstImport.records.map((r) => ({
        question: r.question,
        answer: r.answer,
        category: (r.rawCategory as "consultation") ?? "consultation",
        sanitized: false,
      })),
      { version: "v1.0.1", seed: 42, balanceStrategy: "oversample" }
    );
    const trainQuestions = new Set(
      oversampleRun.samples.filter((s) => s.split === "train").map((s) => s.question)
    );
    const testQuestions = oversampleRun.samples
      .filter((s) => s.split === "test")
      .map((s) => s.question);
    for (const tq of testQuestions) {
      assert.equal(trainQuestions.has(tq), false);
    }

    // 验证脱敏规则（手机号与邮箱均应被替换）
    const sanitizedChunks = run1.chunks.filter((c) => c.sanitized);
    for (const chunk of sanitizedChunks) {
      assert.equal(chunk.content.includes("13812345678"), false);
      assert.equal(chunk.content.includes("help@example.com"), false);
    }

    // 验证数据库落库与读回一致
    const savedSamples = store.listQaSamples(run1.datasetVersion.id);
    assert.equal(savedSamples.length, run1.samples.length);

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
