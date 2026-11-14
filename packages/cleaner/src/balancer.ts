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
 * 先按类别内部确定性切分 train/val/test，再仅对 train 集执行均衡采样（杜绝数据泄露）
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
  const activeCategories = ALL_CATEGORIES.filter((c) => grouped[c].length > 0);
  if (activeCategories.length === 0) {
    throw new Error("没有可用于均衡采样的有效样本");
  }

  // 第一步：在各原始类别内部先完成防泄露的 train / val / test 切分
  const splitBuckets: Record<
    QaCategory,
    { train: CleanedCandidate[]; val: CleanedCandidate[]; test: CleanedCandidate[] }
  > = {
    consultation: { train: [], val: [], test: [] },
    logistics: { train: [], val: [], test: [] },
    refund: { train: [], val: [], test: [] },
    complaint: { train: [], val: [], test: [] },
  };

  activeCategories.forEach((cat, idx) => {
    const shuffled = seededShuffle(grouped[cat], seed + idx * 101);
    const total = shuffled.length;
    const trainEnd = Math.max(1, Math.floor(total * splitRatio.train));
    const valEnd = Math.min(
      total,
      trainEnd + Math.max(total >= 3 ? 1 : 0, Math.floor(total * splitRatio.val))
    );

    splitBuckets[cat] = {
      train: shuffled.slice(0, trainEnd),
      val: shuffled.slice(trainEnd, valEnd),
      test: shuffled.slice(valEnd),
    };
  });

  // 第二步：仅对 train 集执行均衡采样，val 与 test 保持纯净（不复制、不泄露）
  const trainCounts = activeCategories.map((c) => splitBuckets[c].train.length);
  const minTrain = Math.min(...trainCounts);
  const maxTrain = Math.max(...trainCounts);

  const samples: QaSample[] = [];
  const chunks: CleanChunk[] = [];
  const finalDistribution: ClassDistribution = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  let globalIndex = 0;

  for (const cat of activeCategories) {
    const rawTrain = splitBuckets[cat].train;
    let balancedTrain: CleanedCandidate[] = rawTrain;

    if (strategy === "undersample") {
      balancedTrain = rawTrain.slice(0, minTrain);
    } else if (strategy === "oversample") {
      const filled: CleanedCandidate[] = [];
      for (let i = 0; i < maxTrain; i += 1) {
        filled.push(rawTrain[i % rawTrain.length]);
      }
      balancedTrain = filled;
    }

    const pushSample = (
      item: CleanedCandidate,
      split: DatasetSplit,
      idx: number
    ) => {
      const sampleId = `${options.version}-${cat}-${split}-${idx}`;
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

      // 仅将 train 集构建为检索知识块，防止 test 题答案直接泄露进向量库
      if (split === "train") {
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
      }
    };

    balancedTrain.forEach((item, idx) => pushSample(item, "train", idx));
    splitBuckets[cat].val.forEach((item, idx) => pushSample(item, "val", idx));
    splitBuckets[cat].test.forEach((item, idx) => pushSample(item, "test", idx));

    finalDistribution[cat] = balancedTrain.length;
  }

  const datasetVersion: DatasetVersion = {
    id: versionId,
    version: options.version,
    description:
      options.description ??
      `策略=${strategy}, seed=${seed}, 训练集均衡后=${chunks.length}, 总样本=${samples.length}`,
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
