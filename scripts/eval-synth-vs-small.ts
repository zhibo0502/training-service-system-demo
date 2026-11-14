import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { QaCategory, QaSample, StandardCategory } from "@ts-qa/shared";
import { mapToStandardCategory, sanitizeText, seededShuffle } from "../packages/cleaner/src/index.ts";
import { CharNgramNaiveBayesClassifier } from "../packages/trainer/src/index.ts";

interface RawFixtureLine {
  question: string;
  answer: string;
  category?: string;
  rawCategory?: string;
}

const CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

function computeMetricsFromPairs(
  pairs: Array<{ truth: QaCategory; pred: QaCategory }>
) {
  const tp: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  const fp: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  const fn: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  const confusionMatrix: Record<QaCategory, Record<QaCategory, number>> = {
    consultation: { consultation: 0, logistics: 0, refund: 0, complaint: 0 },
    logistics: { consultation: 0, logistics: 0, refund: 0, complaint: 0 },
    refund: { consultation: 0, logistics: 0, refund: 0, complaint: 0 },
    complaint: { consultation: 0, logistics: 0, refund: 0, complaint: 0 },
  };

  let correct = 0;
  for (const { truth, pred } of pairs) {
    confusionMatrix[truth][pred] += 1;
    if (truth === pred) {
      correct += 1;
      tp[truth] += 1;
    } else {
      fp[pred] += 1;
      fn[truth] += 1;
    }
  }

  let f1Sum = 0;
  const perClassF1: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  for (const cat of CATEGORIES) {
    const precision = tp[cat] + fp[cat] > 0 ? tp[cat] / (tp[cat] + fp[cat]) : 0;
    const recall = tp[cat] + fn[cat] > 0 ? tp[cat] / (tp[cat] + fn[cat]) : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;
    perClassF1[cat] = Number(f1.toFixed(4));
    f1Sum += f1;
  }

  return {
    total: pairs.length,
    correct,
    accuracy: Number((correct / pairs.length).toFixed(4)),
    macroF1: Number((f1Sum / CATEGORIES.length).toFixed(4)),
    perClassF1,
    confusionMatrix,
  };
}

function measureDiversity(questions: readonly string[]) {
  const unigrams = new Set<string>();
  const bigrams = new Set<string>();
  let totalLen = 0;

  for (const q of questions) {
    const norm = q.replace(/\s+/g, "");
    totalLen += norm.length;
    for (let i = 0; i < norm.length; i += 1) {
      unigrams.add(norm[i]);
      if (i + 1 < norm.length) {
        bigrams.add(norm.slice(i, i + 2));
      }
    }
  }

  const meanLen = totalLen / questions.length;
  const variance =
    questions.reduce((acc, q) => {
      const diff = q.replace(/\s+/g, "").length - meanLen;
      return acc + diff * diff;
    }, 0) / questions.length;

  return {
    count: questions.length,
    uniqueUnigrams: unigrams.size,
    uniqueBigrams: bigrams.size,
    avgCharLength: Number(meanLen.toFixed(2)),
    stdCharLength: Number(Math.sqrt(variance).toFixed(2)),
  };
}

function loadJsonlSamples(
  filePath: string,
  datasetVersionId: string
): QaSample[] {
  const lines = readFileSync(filePath, "utf-8")
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  return lines.map((line, idx) => {
    const raw = JSON.parse(line) as RawFixtureLine;
    const cleanedQ = sanitizeText(raw.question).cleanedText;
    const cleanedA = sanitizeText(raw.answer).cleanedText;
    const cat = mapToStandardCategory(
      raw.category ?? raw.rawCategory,
      cleanedQ
    ) as StandardCategory;
    return {
      id: `${datasetVersionId}-${idx}`,
      datasetVersionId,
      question: cleanedQ,
      answer: cleanedA,
      category: cat,
      split: "train",
    };
  });
}

async function main() {
  const repoRoot = resolve(import.meta.dirname, "..");
  const realPath = resolve(
    repoRoot,
    "packages/cleaner/test/fixtures/common-100-qa.jsonl"
  );
  const base10kPath = resolve(repoRoot, "data/corpus-10k.jsonl");
  const synth100kPath = resolve(repoRoot, "data/synthetic-100k.jsonl");
  const reportPath = resolve(repoRoot, "data/eval-synth-vs-small-report.json");

  const realSamples = seededShuffle(
    loadJsonlSamples(realPath, "real-100"),
    2026
  );
  const base10kSamples = loadJsonlSamples(base10kPath, "base-10k");
  const synth100kSamples = loadJsonlSamples(synth100kPath, "synth-100k");

  // 1. 内在多样性对比
  const diversity = {
    smallReal100: measureDiversity(realSamples.map((s) => s.question)),
    baseSynth10k: measureDiversity(base10kSamples.map((s) => s.question)),
    fullSynth100k: measureDiversity(synth100kSamples.map((s) => s.question)),
  };

  // 分层构造 5 折交叉验证（保证每折 20 条四类均衡）
  const folds: QaSample[][] = [[], [], [], [], []];
  for (const cat of CATEGORIES) {
    const catItems = realSamples.filter((s) => s.category === cat);
    catItems.forEach((item, idx) => {
      folds[idx % 5].push(item);
    });
  }

  // 组 A：仅小数据集基线（5-fold CV，每折 80 条训练，测 20 条真实样本）
  const smallCvPairs: Array<{ truth: QaCategory; pred: QaCategory }> = [];
  const smallErrors: Array<{ question: string; truth: string; pred: string }> = [];
  for (let f = 0; f < 5; f += 1) {
    const testFold = folds[f];
    const trainFold = folds.flatMap((fold, idx) => (idx === f ? [] : fold));
    const clf = new CharNgramNaiveBayesClassifier({
      ngramMin: 1,
      ngramMax: 2,
      alpha: 0.5,
    });
    clf.fit(trainFold);
    for (const item of testFold) {
      const pred = clf.predict(item.question);
      smallCvPairs.push({ truth: item.category, pred });
      if (pred !== item.category) {
        smallErrors.push({
          question: item.question,
          truth: item.category,
          pred,
        });
      }
    }
  }
  const smallMetrics = computeMetricsFromPairs(smallCvPairs);

  // 组 B：纯 100k 合成数据集（零泄漏，全量 100,000 条训练，直接测 100 条外部真实数据）
  const synthClf = new CharNgramNaiveBayesClassifier({
    ngramMin: 1,
    ngramMax: 2,
    alpha: 0.5,
  });
  synthClf.fit(synth100kSamples);
  const synthPairs: Array<{ truth: QaCategory; pred: QaCategory }> = [];
  const synthErrors: Array<{ question: string; truth: string; pred: string }> = [];
  for (const item of realSamples) {
    const pred = synthClf.predict(item.question);
    synthPairs.push({ truth: item.category, pred });
    if (pred !== item.category) {
      synthErrors.push({
        question: item.question,
        truth: item.category,
        pred,
      });
    }
  }
  const synthMetrics = computeMetricsFromPairs(synthPairs);

  // 组 C：真实小数据（过采样加权 x100）+ 100k 合成数据混合（5-fold CV）
  const hybridPairs: Array<{ truth: QaCategory; pred: QaCategory }> = [];
  const hybridErrors: Array<{ question: string; truth: string; pred: string }> = [];
  for (let f = 0; f < 5; f += 1) {
    const testFold = folds[f];
    const trainFold = folds.flatMap((fold, idx) => (idx === f ? [] : fold));
    const upsampledReal: QaSample[] = [];
    for (let r = 0; r < 100; r += 1) {
      upsampledReal.push(...trainFold);
    }
    const clf = new CharNgramNaiveBayesClassifier({
      ngramMin: 1,
      ngramMax: 2,
      alpha: 0.5,
    });
    clf.fit([...synth100kSamples, ...upsampledReal]);
    for (const item of testFold) {
      const pred = clf.predict(item.question);
      hybridPairs.push({ truth: item.category, pred });
      if (pred !== item.category) {
        hybridErrors.push({
          question: item.question,
          truth: item.category,
          pred,
        });
      }
    }
  }
  const hybridMetrics = computeMetricsFromPairs(hybridPairs);

  const report = {
    benchmarkDataset: "packages/cleaner/test/fixtures/common-100-qa.jsonl (100 条外部真实问法)",
    diversityComparison: diversity,
    resultsOnExternalReal100: {
      A_smallRealOnly_5FoldCV: smallMetrics,
      B_pureSynthetic100k_ZeroShotOnReal: synthMetrics,
      C_hybridSynth100kPlusRealUpsampled_5FoldCV: hybridMetrics,
    },
    sampleErrors: {
      smallOnlyErrors: smallErrors.slice(0, 10),
      pureSynth100kErrors: synthErrors.slice(0, 10),
      hybridErrors: hybridErrors.slice(0, 10),
    },
  };

  writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf-8");

  console.log("=== 内在多样性对比 ===");
  console.table(diversity);
  console.log("\n=== 外部真实 100 条问法盲测对比 ===");
  console.table({
    "A. 仅小数据集 (80条 5-fold CV)": {
      Accuracy: smallMetrics.accuracy,
      "Macro-F1": smallMetrics.macroF1,
      errors: smallErrors.length,
    },
    "B. 纯 100k 合成数据 (零真实训练)": {
      Accuracy: synthMetrics.accuracy,
      "Macro-F1": synthMetrics.macroF1,
      errors: synthErrors.length,
    },
    "C. 混合 (100k合成 + 真实小数据加权)": {
      Accuracy: hybridMetrics.accuracy,
      "Macro-F1": hybridMetrics.macroF1,
      errors: hybridErrors.length,
    },
  });
  console.log(`\n详细报告已写入: ${reportPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
