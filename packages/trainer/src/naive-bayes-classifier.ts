import type { QaCategory, QaSample } from "@ts-qa/shared";

const CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

export interface NaiveBayesOptions {
  ngramMin?: number;
  ngramMax?: number;
  alpha?: number;
}

/**
 * 纯 TypeScript 中文字符 N-gram 多项式朴素贝叶斯分类器（从 train 集真实学习）
 */
export class CharNgramNaiveBayesClassifier {
  private readonly ngramMin: number;
  private readonly ngramMax: number;
  private readonly alpha: number;

  private readonly vocab = new Set<string>();
  private readonly classTokenCounts: Record<QaCategory, Map<string, number>> = {
    consultation: new Map(),
    logistics: new Map(),
    refund: new Map(),
    complaint: new Map(),
  };
  private readonly classTotalTokens: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  private readonly classDocCounts: Record<QaCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  private totalDocs = 0;

  constructor(options?: NaiveBayesOptions) {
    this.ngramMin = options?.ngramMin ?? 1;
    this.ngramMax = options?.ngramMax ?? 2;
    this.alpha = options?.alpha ?? 1.0;
  }

  fit(trainSamples: readonly QaSample[]): void {
    this.vocab.clear();
    this.totalDocs = 0;
    for (const cat of CATEGORIES) {
      this.classTokenCounts[cat].clear();
      this.classTotalTokens[cat] = 0;
      this.classDocCounts[cat] = 0;
    }

    for (const sample of trainSamples) {
      const tokens = this.extractNgrams(sample.question);
      this.classDocCounts[sample.category] += 1;
      this.totalDocs += 1;

      for (const tok of tokens) {
        this.vocab.add(tok);
        const prev = this.classTokenCounts[sample.category].get(tok) ?? 0;
        this.classTokenCounts[sample.category].set(tok, prev + 1);
        this.classTotalTokens[sample.category] += 1;
      }
    }
  }

  predict(text: string): QaCategory {
    if (this.totalDocs === 0) {
      return "consultation";
    }

    const tokens = this.extractNgrams(text);
    const vocabSize = Math.max(1, this.vocab.size);
    let bestCat: QaCategory = "consultation";
    let bestLogProb = -Infinity;

    for (const cat of CATEGORIES) {
      const docCount = this.classDocCounts[cat];
      if (docCount === 0) continue;

      let logProb = Math.log(docCount / this.totalDocs);
      const denom = this.classTotalTokens[cat] + this.alpha * vocabSize;

      for (const tok of tokens) {
        const count = this.classTokenCounts[cat].get(tok) ?? 0;
        logProb += Math.log((count + this.alpha) / denom);
      }

      if (logProb > bestLogProb) {
        bestLogProb = logProb;
        bestCat = cat;
      }
    }

    return bestCat;
  }

  private extractNgrams(text: string): string[] {
    const cleaned = text.replace(/\s+/g, "").toLowerCase();
    const grams: string[] = [];
    for (let n = this.ngramMin; n <= this.ngramMax; n += 1) {
      for (let i = 0; i <= cleaned.length - n; i += 1) {
        grams.push(cleaned.slice(i, i + n));
      }
    }
    return grams;
  }
}
