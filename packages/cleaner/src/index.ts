import type { RawCorpusRecord, SqliteStore } from "@ts-qa/collector";
import { mapToStandardCategory } from "./category-mapper.ts";
import { sanitizeText } from "./sanitizer.ts";
import {
  balanceAndSplitSamples,
  type BalanceOptions,
  type BalancePipelineOutput,
  type CleanedCandidate,
} from "./balancer.ts";

export * from "./prng.ts";
export * from "./sanitizer.ts";
export * from "./category-mapper.ts";
export * from "./balancer.ts";

/**
 * 端到端数据清洗与均衡落库流水线
 */
export function runCleaningPipeline(
  records: readonly RawCorpusRecord[],
  documentId: string,
  store: SqliteStore,
  options: BalanceOptions
): BalancePipelineOutput {
  const seenQuestions = new Set<string>();
  const candidates: CleanedCandidate[] = [];

  for (const record of records) {
    const q = sanitizeText(record.question);
    const a = sanitizeText(record.answer);

    if (q.cleanedText.length < 2 || a.cleanedText.length < 2) {
      continue;
    }

    const dedupeKey = q.cleanedText.toLowerCase();
    if (seenQuestions.has(dedupeKey)) {
      continue;
    }
    seenQuestions.add(dedupeKey);

    candidates.push({
      question: q.cleanedText,
      answer: a.cleanedText,
      category: mapToStandardCategory(record.rawCategory, q.cleanedText),
      sanitized: q.sanitized || a.sanitized,
      intentTag: record.intentTag,
      sourceRefId: documentId,
    });
  }

  const output = balanceAndSplitSamples(candidates, options);
  store.saveDatasetVersion(output.datasetVersion);
  store.saveQaSamples(output.samples);
  store.saveCleanChunks(output.chunks);

  return output;
}
