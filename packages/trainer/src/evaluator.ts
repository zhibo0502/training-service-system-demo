import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type {
  EvaluationMetrics,
  QaCategory,
  QaSample,
} from "@ts-qa/shared";

const CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

/**
 * 计算分类器在均衡测试集上的 Accuracy、Macro-F1 与各类别召回率
 */
export async function evaluateIntentClassifier(
  testSamples: readonly QaSample[],
  predictCategory: (question: string) => Promise<QaCategory> | QaCategory
): Promise<EvaluationMetrics> {
  if (testSamples.length === 0) {
    throw new Error("评测集为空，无法计算指标");
  }

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

  for (const sample of testSamples) {
    const predicted = await predictCategory(sample.question);
    confusionMatrix[sample.category][predicted] += 1;
    if (predicted === sample.category) {
      correct += 1;
      tp[sample.category] += 1;
    } else {
      fp[predicted] += 1;
      fn[sample.category] += 1;
    }
  }

  const perClassRecall: Partial<Record<QaCategory, number>> = {};
  const perClassF1: Partial<Record<QaCategory, number>> = {};
  let f1Sum = 0;
  let activeClasses = 0;

  for (const cat of CATEGORIES) {
    const actualCount = tp[cat] + fn[cat];
    if (actualCount === 0 && fp[cat] === 0) continue;

    activeClasses += 1;
    const precision = tp[cat] + fp[cat] > 0 ? tp[cat] / (tp[cat] + fp[cat]) : 0;
    const recall = actualCount > 0 ? tp[cat] / actualCount : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;

    perClassRecall[cat] = Number(recall.toFixed(4));
    perClassF1[cat] = Number(f1.toFixed(4));
    f1Sum += f1;
  }

  return {
    accuracy: Number((correct / testSamples.length).toFixed(4)),
    macroF1:
      activeClasses > 0 ? Number((f1Sum / activeClasses).toFixed(4)) : 0,
    perClassRecall,
    perClassF1,
    confusionMatrix,
    evaluatedCount: testSamples.length,
  };
}

/**
 * 将均衡样本导出为标准指令微调 JSONL 文件（用于二期 QLoRA 训练）
 */
export function exportFinetuneJsonl(
  samples: readonly QaSample[],
  outputPath: string
): number {
  mkdirSync(dirname(outputPath), { recursive: true });
  const lines = samples.map((s) =>
    JSON.stringify({
      instruction: `请作为专业中文客服回答以下【${s.category}】类客户问题：`,
      input: s.question,
      output: s.answer,
      category: s.category,
      split: s.split ?? "train",
    })
  );
  writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf-8");
  return lines.length;
}
