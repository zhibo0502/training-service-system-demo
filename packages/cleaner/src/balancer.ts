import { randomUUID } from "node:crypto";
import type {
  BalanceStrategy,
  ClassDistribution,
  CleanChunk,
  DatasetSplit,
  DatasetVersion,
  QaCategory,
  QaSample,
} from "@ts-qa/shared";
import { seededShuffle } from "./prng.ts";

export interface CleanedCandidate {
  question: string;
  answer: string;
  category: QaCategory;
  sanitized: boolean;
  intentTag?: string;
  sourceRefId?: string;
}

export interface BalanceOptions {
  version: string;
  description?: string;
  seed?: number;
  balanceStrategy?: BalanceStrategy;
  splitRatio?: {
    train: number;
    val: number;
    test: number;
  };
  artifactDir?: string;
}

export interface BalancePipelineOutput {
  datasetVersion: DatasetVersion;
  rawDistribution: ClassDistribution;
  samples: QaSample[];
  chunks: CleanChunk[];
}

const ALL_CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

/**
 * 执行样本均衡采样、确定性 train/val/test 切分并生成版本快照
 */
export function balanceAndSplitSamples(
  candidates: readonly CleanedCandidate[],
  options: BalanceOptions
): BalancePipelineOutput {
  const seed = options.seed ?? 42;
  const strategy: BalanceStrategy = options.balanceStrategy ?? "undersample";
  const splitRatio = options.splitRatio ?? { train: 0.8, val: 0.1, test: 0.1 };
  const now = new Date().toISOString();
  const versionId = randomUUID();

  const grouped: Record<QaCategory, CleanedCandidate[]> = {
    consultation: [],
    logistics: [],
    refund: [],
    complaint: [],
  };

  for (const item of candidates) {
    grouped[item.category].push(item);
  }

  const rawDistribution = countDistribution(grouped);
  const counts = ALL_CATEGORIES.map((c) => grouped[c].length).filter((n) => n > 0);
  if (counts.length === 0) {
    throw new Error("没有可用于均衡采样的有效样本");
  }

  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);

  const balancedByCategory: Record<QaCategory, CleanedCandidate[]> = {
    consultation: [],
    logistics: [],
    refund: [],
    complaint: [],
  };

  ALL_CATEGORIES.forEach((cat, idx) => {
    const items = seededShuffle(grouped[cat], seed + idx * 101);
    if (items.length === 0) return;

    if (strategy === "undersample") {
      balancedByCategory[cat] = items.slice(0, minCount);
    } else if (strategy === "oversample") {
      const filled: CleanedCandidate[] = [];
      for (let i = 0; i < maxCount; i += 1) {
        filled.push(items[i % items.length]);
      }
      balancedByCategory[cat] = filled;
    } else {
      balancedByCategory[cat] = items;
    }
  });

  const samples: QaSample[] = [];
  const chunks: CleanChunk[] = [];
  let globalIndex = 0;

  for (const cat of ALL_CATEGORIES) {
    const catItems = balancedByCategory[cat];
    const total = catItems.length;
    const trainEnd = Math.max(1, Math.floor(total * splitRatio.train));
    const valEnd = Math.min(
      total,
      trainEnd + Math.max(total >= 3 ? 1 : 0, Math.floor(total * splitRatio.val))
    );

    catItems.forEach((item, idx) => {
      let split: DatasetSplit = "test";
      if (idx < trainEnd) {
        split = "train";
      } else if (idx < valEnd) {
        split = "val";
      }

      const sampleId = `${options.version}-${cat}-${idx}`;
      samples.push({
        id: sampleId,
        question: item.question,
        answer: item.answer,
        category: cat,
        split,
        datasetVersionId: versionId,
        source: "chat_log_extracted",
        sourceRefId: item.sourceRefId,
        intentTag: item.intentTag,
        qualityScore: 1,
        reviewStatus: "approved",
        createdAt: now,
        updatedAt: now,
      });

      const chunkContent = `问题：${item.question}\n标准回答：${item.answer}`;
      chunks.push({
        id: `${sampleId}-chunk`,
        documentId: item.sourceRefId ?? versionId,
        chunkIndex: globalIndex,
        content: chunkContent,
        charCount: chunkContent.length,
        tokenEstimate: Math.ceil(chunkContent.length / 1.5),
        sanitized: item.sanitized,
        metadata: {
          sourceLabel: cat,
        },
        vectorStatus: "pending",
        createdAt: now,
      });

      globalIndex += 1;
    });
  }

  const finalDistribution = countDistribution(balancedByCategory);
  const datasetVersion: DatasetVersion = {
    id: versionId,
    version: options.version,
    description:
      options.description ??
      `策略=${strategy}, seed=${seed}, 均衡后总样本=${samples.length}`,
    seed,
    balanceStrategy: strategy,
    splitRatio,
    totalCount: samples.length,
    classDistribution: finalDistribution,
    artifactDir: options.artifactDir ?? `data/versions/${options.version}`,
    createdAt: now,
  };

  return {
    datasetVersion,
    rawDistribution,
    samples,
    chunks,
  };
}

function countDistribution(
  grouped: Record<QaCategory, unknown[]>
): ClassDistribution {
  return {
    consultation: grouped.consultation.length,
    logistics: grouped.logistics.length,
    refund: grouped.refund.length,
    complaint: grouped.complaint.length,
  };
}
