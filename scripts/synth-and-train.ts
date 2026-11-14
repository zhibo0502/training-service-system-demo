import { createWriteStream, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { QaSample, StandardCategory } from "@ts-qa/shared";
import { createSeededRng, synthesizeQaDataset } from "../packages/cleaner/src/index.ts";
import {
  CharNgramNaiveBayesClassifier,
  evaluateIntentClassifier,
} from "../packages/trainer/src/index.ts";

async function writeJsonlStream(
  filePath: string,
  items: readonly unknown[],
  mapLine: (item: unknown) => string
): Promise<void> {
  await new Promise<void>((resolvePromise, reject) => {
    const stream = createWriteStream(filePath, { encoding: "utf-8" });
    stream.on("error", reject);
    stream.on("finish", () => resolvePromise());
    for (const item of items) {
      stream.write(`${mapLine(item)}\n`);
    }
    stream.end();
  });
}

async function main() {
  const repoRoot = resolve(import.meta.dirname, "..");
  const dataDir = resolve(repoRoot, "data");
  mkdirSync(dataDir, { recursive: true });

  const seed = Number(process.env.SYNTH_SEED ?? 20261114);
  const baseCount = Number(process.env.SYNTH_BASE_COUNT ?? 10_000);
  const targetCount = Number(process.env.SYNTH_TARGET_COUNT ?? 100_000);

  console.log(
    `[1/4] 正在生成基础语料 (${baseCount} 条) 并合成扩增至 ${targetCount} 条 (seed=${seed})...`
  );
  const synth = synthesizeQaDataset({ baseCount, targetCount, seed });
  console.log(
    `      完成！耗时 ${synth.stats.durationMs}ms，类别分布：`,
    synth.stats.categoryCounts
  );

  const basePath = resolve(dataDir, "corpus-10k.jsonl");
  const synthPath = resolve(dataDir, "synthetic-100k.jsonl");
  const finetunePath = resolve(dataDir, "finetune-100k.jsonl");
  const reportPath = resolve(dataDir, "synth-100k-report.json");

  console.log(`[2/4] 正在写入 JSONL 语料文件与 SFT 微调集...`);
  await writeJsonlStream(basePath, synth.baseRecords, (r) => JSON.stringify(r));
  await writeJsonlStream(synthPath, synth.allRecords, (r) => JSON.stringify(r));
  await writeJsonlStream(finetunePath, synth.allRecords, (raw) => {
    const r = raw as (typeof synth.allRecords)[number];
    return JSON.stringify({
      messages: [
        {
          role: "system",
          content: "你是一个严谨、专业的离线电商客服助手，请依据业务规则准确作答。",
        },
        { role: "user", content: r.question },
        { role: "assistant", content: r.answer },
      ],
      metadata: {
        category: r.rawCategory,
        intentTag: r.intentTag,
        sourceType: r.sourceType,
        baseKnowledgeId: r.baseKnowledgeId,
      },
    });
  });

  console.log(
    `[3/4] 正在按母本 baseKnowledgeId 分组切分 (80% train / 10% val / 10% test，防止同义变体泄漏)...`
  );
  const rng = createSeededRng(seed + 1);
  const groupsByCat: Record<StandardCategory, string[]> = {
    consultation: [],
    logistics: [],
    refund: [],
    complaint: [],
  };
  for (const base of synth.baseRecords) {
    groupsByCat[base.rawCategory].push(base.baseKnowledgeId);
  }

  const groupSplitMap = new Map<string, "train" | "val" | "test">();
  for (const cat of Object.keys(groupsByCat) as StandardCategory[]) {
    const ids = groupsByCat[cat];
    for (let i = ids.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = ids[i];
      ids[i] = ids[j];
      ids[j] = tmp;
    }
    const trainEnd = Math.floor(ids.length * 0.8);
    const valEnd = Math.floor(ids.length * 0.9);
    for (let i = 0; i < ids.length; i += 1) {
      groupSplitMap.set(
        ids[i],
        i < trainEnd ? "train" : i < valEnd ? "val" : "test"
      );
    }
  }

  const trainSamples: QaSample[] = [];
  const valSamples: QaSample[] = [];
  const testSamples: QaSample[] = [];

  for (const r of synth.allRecords) {
    const split = groupSplitMap.get(r.baseKnowledgeId) ?? "train";
    const sample: QaSample = {
      id: r.id,
      datasetVersionId: "synth-100k",
      question: r.question,
      answer: r.answer,
      category: r.rawCategory,
      split,
      sourceRefId: r.baseKnowledgeId,
    };
    if (split === "train") trainSamples.push(sample);
    else if (split === "val") valSamples.push(sample);
    else testSamples.push(sample);
  }

  console.log(
    `      切分完成：train=${trainSamples.length}, val=${valSamples.length}, test=${testSamples.length}`
  );

  console.log(`[4/4] 正在训练意图分类器并在独立 test 集评测...`);
  const trainStart = Date.now();
  const clf = new CharNgramNaiveBayesClassifier({
    ngramMin: 1,
    ngramMax: 2,
    alpha: 0.5,
  });
  clf.fit(trainSamples);
  const trainMs = Date.now() - trainStart;

  const evalStart = Date.now();
  const metrics = await evaluateIntentClassifier(testSamples, (q) =>
    clf.predict(q)
  );
  const evalMs = Date.now() - evalStart;

  const report = {
    seed,
    files: {
      baseJsonl: basePath,
      syntheticJsonl: synthPath,
      finetuneJsonl: finetunePath,
    },
    datasetStats: synth.stats,
    leakFreeSplit: {
      rule: "grouped_by_baseKnowledgeId",
      trainCount: trainSamples.length,
      valCount: valSamples.length,
      testCount: testSamples.length,
    },
    ragNote:
      "RAG 向量库建议仅索引 corpus-10k.jsonl 中的 10,000 条去重母本知识，避免 100,000 条同义变体挤占 Top-K。",
    classifierBenchmark: {
      model: "CharNgramNaiveBayes(n=1..2, alpha=0.5)",
      trainMs,
      evalMs,
      metrics,
    },
  };

  writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf-8");
  console.log(`\n全部完成！训练耗时 ${trainMs}ms，评测耗时 ${evalMs}ms`);
  console.log(
    `独立 Test 集评测 (${testSamples.length} 条未见过母本): Accuracy=${metrics.accuracy}, Macro-F1=${metrics.macroF1}`
  );
  console.log(`报告已保存至: ${reportPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
