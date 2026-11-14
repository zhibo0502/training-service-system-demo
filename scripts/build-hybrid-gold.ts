import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { QaCategory, QaSample, StandardCategory } from "@ts-qa/shared";
import {
  mapToStandardCategory,
  sanitizeText,
  seededShuffle,
} from "../packages/cleaner/src/index.ts";
import { CharNgramNaiveBayesClassifier } from "../packages/trainer/src/index.ts";

interface RawLine {
  question: string;
  answer: string;
  category?: string;
  rawCategory?: string;
  source?: string;
}

interface HybridRecord {
  question: string;
  answer: string;
  category: QaCategory;
  source: string;
}

const CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

function normalizeForDedup(text: string): string {
  return sanitizeText(text)
    .cleanedText
    .toLowerCase()
    .replace(/[\s\p{P}\p{S}]+/gu, "");
}

function resolveCategory(row: RawLine): QaCategory {
  const raw = row.category ?? row.rawCategory ?? "";
  if (CATEGORIES.includes(raw as QaCategory)) {
    return raw as QaCategory;
  }
  return mapToStandardCategory(raw, row.question, row.answer) as QaCategory;
}

function loadJsonl(filePath: string, defaultSource: string): HybridRecord[] {
  const raw = readFileSync(filePath, "utf8");
  const out: HybridRecord[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const obj = JSON.parse(trimmed) as RawLine;
    const q = sanitizeText(obj.question ?? "").cleanedText;
    const a = sanitizeText(obj.answer ?? "").cleanedText;
    if (!q || !a) continue;
    out.push({
      question: q,
      answer: a,
      category: resolveCategory(obj),
      source: obj.source ?? defaultSource,
    });
  }
  return out;
}

function stratifiedSample(
  records: HybridRecord[],
  perCategoryLimit: number,
  seed: number
): HybridRecord[] {
  const byCat = new Map<QaCategory, HybridRecord[]>();
  for (const cat of CATEGORIES) byCat.set(cat, []);
  for (const r of records) {
    byCat.get(r.category)?.push(r);
  }
  const picked: HybridRecord[] = [];
  for (let i = 0; i < CATEGORIES.length; i++) {
    const cat = CATEGORIES[i];
    const list = seededShuffle(byCat.get(cat) ?? [], seed + i * 31);
    picked.push(...list.slice(0, perCategoryLimit));
  }
  return picked;
}

function toQaSamples(
  records: HybridRecord[],
  split: "train" | "test"
): QaSample[] {
  return records.map((r, idx) => ({
    id: `${split}-${idx}`,
    datasetVersionId: "eval",
    question: r.question,
    answer: r.answer,
    category: r.category as StandardCategory,
    split,
    sourceDocId: r.source,
    keywords: [],
  }));
}

function computeMetrics(
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
  let correct = 0;
  for (const { truth, pred } of pairs) {
    if (truth === pred) {
      correct += 1;
      tp[truth] += 1;
    } else {
      fp[pred] += 1;
      fn[truth] += 1;
    }
  }
  let f1Sum = 0;
  for (const cat of CATEGORIES) {
    const p = tp[cat] + fp[cat] > 0 ? tp[cat] / (tp[cat] + fp[cat]) : 0;
    const r = tp[cat] + fn[cat] > 0 ? tp[cat] / (tp[cat] + fn[cat]) : 0;
    f1Sum += p + r > 0 ? (2 * p * r) / (p + r) : 0;
  }
  return {
    total: pairs.length,
    accuracy: pairs.length > 0 ? Number((correct / pairs.length).toFixed(4)) : 0,
    macroF1: Number((f1Sum / CATEGORIES.length).toFixed(4)),
  };
}

function evalClassifier(
  trainRecords: HybridRecord[],
  evalRecords: HybridRecord[],
  ngramMax = 2,
  alpha = 0.5
) {
  const clf = new CharNgramNaiveBayesClassifier({ ngramMax, alpha });
  clf.fit(toQaSamples(trainRecords, "train"));
  const pairs = evalRecords.map((r) => ({
    truth: r.category,
    pred: clf.predict(r.question),
  }));
  return computeMetrics(pairs);
}

async function main() {
  const root = resolve(import.meta.dirname, "..");

  const common100 = loadJsonl(
    resolve(root, "packages/cleaner/test/fixtures/common-100-qa.jsonl"),
    "common-100-qa"
  );
  const taobao = loadJsonl(
    resolve(root, "data/external/taobao-ecd-qa.jsonl"),
    "taobao-ecd-qa"
  );
  const openReal2k = loadJsonl(
    resolve(root, "data/external/open-real-2k-sample.jsonl"),
    "open-real-2k"
  );
  const bayes8k = loadJsonl(
    resolve(root, "data/bayes-enhanced-train-8k.jsonl"),
    "bayes-enhanced-8k"
  );
  const corpus10k = loadJsonl(
    resolve(root, "data/corpus-10k.jsonl"),
    "corpus-10k"
  );
  const blindHardEval = loadJsonl(
    resolve(root, "data/bayes-hard-eval-200.jsonl"),
    "bayes-hard-eval"
  );

  // Build strict exclusion set from blindHardEval to prevent any data leakage
  const blindNormSet = new Set(
    blindHardEval.map((r) => normalizeForDedup(r.question))
  );

  const sampledBayes = stratifiedSample(bayes8k, 600, 42);
  const sampled10k = stratifiedSample(corpus10k, 400, 99);

  // Priority order: real human-verified & real e-commerce first, then Bayes hard, then 10k slots
  const rawMerged: HybridRecord[] = [
    ...common100.map((r) => ({ ...r, source: "common-100-qa" })),
    ...taobao.map((r) => ({ ...r, source: "taobao-ecd-qa" })),
    ...openReal2k.map((r) => ({ ...r, source: "open-real-2k" })),
    ...sampledBayes.map((r) => ({ ...r, source: "bayes-enhanced-8k" })),
    ...sampled10k.map((r) => ({ ...r, source: "corpus-10k" })),
  ];

  const seenNorm = new Set<string>();
  const hybridGold: HybridRecord[] = [];
  let skippedDup = 0;
  let skippedLeak = 0;

  for (const r of rawMerged) {
    const norm = normalizeForDedup(r.question);
    if (!norm) continue;
    if (blindNormSet.has(norm)) {
      skippedLeak += 1;
      continue;
    }
    if (seenNorm.has(norm)) {
      skippedDup += 1;
      continue;
    }
    seenNorm.add(norm);
    hybridGold.push(r);
  }

  // Shuffle deterministically so records are well-mixed
  const finalHybrid = seededShuffle(hybridGold, 2026);
  const outPath = resolve(root, "data/hybrid-gold-train.jsonl");
  writeFileSync(
    outPath,
    finalHybrid.map((r) => JSON.stringify(r)).join("\n") + "\n",
    "utf8"
  );

  const byCat: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  const bySrc: Record<string, number> = {};
  for (const r of finalHybrid) {
    byCat[r.category] += 1;
    bySrc[r.source] = (bySrc[r.source] ?? 0) + 1;
  }

  // Prepare unseen real eval set from open-real-41k.jsonl (400 samples not in seenNorm or blindNormSet)
  const open41kPath = resolve(root, "data/external/open-real-41k.jsonl");
  let unseenRealEval: HybridRecord[] = [];
  if (existsSync(open41kPath)) {
    const all41k = loadJsonl(open41kPath, "open-real-41k").filter((r) => {
      const n = normalizeForDedup(r.question);
      return n.length >= 4 && !seenNorm.has(n) && !blindNormSet.has(n);
    });
    unseenRealEval = stratifiedSample(all41k, 100, 777);
  }

  // Also create an internal 10% held-out split of hybridGold to verify in-domain fit
  const splitIdx = Math.floor(finalHybrid.length * 0.9);
  const hybridTrainSplit = finalHybrid.slice(0, splitIdx);
  const hybridTestSplit = finalHybrid.slice(splitIdx);

  const report = {
    outputPath: outPath,
    totalRecords: finalHybrid.length,
    skippedDuplicates: skippedDup,
    skippedBlindLeakage: skippedLeak,
    categoryCounts: byCat,
    sourceCounts: bySrc,
    evaluations: {
      blindHardEval158: {
        common100Only: evalClassifier(common100, blindHardEval),
        openReal2kOnly: evalClassifier(openReal2k, blindHardEval),
        corpus10kOnly: evalClassifier(corpus10k, blindHardEval),
        hybridGold: evalClassifier(finalHybrid, blindHardEval),
      },
      unseenRealEval: unseenRealEval.length
        ? {
            sampleCount: unseenRealEval.length,
            common100Only: evalClassifier(common100, unseenRealEval),
            openReal2kOnly: evalClassifier(openReal2k, unseenRealEval),
            corpus10kOnly: evalClassifier(corpus10k, unseenRealEval),
            hybridGold: evalClassifier(finalHybrid, unseenRealEval),
          }
        : null,
      hybridInDomainHeldout10Pct: evalClassifier(
        hybridTrainSplit,
        hybridTestSplit
      ),
    },
  };

  console.log(JSON.stringify(report, null, 2));
}

main();
