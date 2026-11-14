import assert from "node:assert/strict";
import test from "node:test";
import { synthesizeQaDataset } from "../src/index.ts";

test("确定性合成引擎：可复现、严格去重且类别均衡", () => {
  const res1 = synthesizeQaDataset({
    baseCount: 200,
    targetCount: 1000,
    seed: 2026,
  });
  const res2 = synthesizeQaDataset({
    baseCount: 200,
    targetCount: 1000,
    seed: 2026,
  });

  assert.equal(res1.baseRecords.length, 200);
  assert.equal(res1.allRecords.length, 1000);
  assert.deepEqual(res1.allRecords, res2.allRecords);

  // 严格去重校验
  const uniqueQuestions = new Set(
    res1.allRecords.map((r) => r.question.toLowerCase())
  );
  assert.equal(uniqueQuestions.size, 1000);

  // 四大类严格均衡（各 250 条）
  assert.equal(res1.stats.categoryCounts.consultation, 250);
  assert.equal(res1.stats.categoryCounts.logistics, 250);
  assert.equal(res1.stats.categoryCounts.refund, 250);
  assert.equal(res1.stats.categoryCounts.complaint, 250);

  // 所有合成记录都能追溯到有效母本
  const baseIds = new Set(res1.baseRecords.map((b) => b.baseKnowledgeId));
  for (const r of res1.allRecords) {
    assert.ok(baseIds.has(r.baseKnowledgeId));
  }
});
