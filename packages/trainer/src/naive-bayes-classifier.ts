import type { QaCategory, QaSample } from "@ts-qa/shared";

const CATEGORIES: readonly QaCategory[] = [
  "consultation",
  "logistics",
  "refund",
  "complaint",
];

const STOP_CHARS = new Set([
  "的", "了", "吗", "呢", "啊", "吧", "呀", "么", "这", "那",
  "我", "你", "您", "他", "她", "们", "在", "是", "有", "和",
  "与", "及", "还", "能", "会", "要", "都", "就", "也", "很",
]);

export interface NaiveBayesOptions {
  ngramMin?: number;
  ngramMax?: number;
  alpha?: number;
  bigramWeight?: number;
}

/**
 * 纯 TypeScript 中文字符 N-gram 朴素贝叶斯分类器（含停用字过滤、IDF 加权与词组权重）
 */
export class CharNgramNaiveBayesClassifier {
  private readonly ngramMin: number;
  private readonly ngramMax: number;
  private readonly alpha: number;
  private readonly bigramWeight: number;

  private readonly vocab = new Set<string>();
  private readonly tokenIdf = new Map<string, number>();
  private readonly classTokenWeights: Record<QaCategory, Map<string, number>> = {
    consultation: new Map(),
    logistics: new Map(),
    refund: new Map(),
    complaint: new Map(),
  };
  private readonly classTotalWeight: Record<QaCategory, number> = {
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
    this.bigramWeight = options?.bigramWeight ?? 2.5;
  }

  fit(trainSamples: readonly QaSample[]): void {
    this.vocab.clear();
    this.tokenIdf.clear();
    this.totalDocs = 0;
    for (const cat of CATEGORIES) {
      this.classTokenWeights[cat].clear();
      this.classTotalWeight[cat] = 0;
      this.classDocCounts[cat] = 0;
    }

    // 先统计每个特征的文档频率，计算 IDF 以抑制跨类高频通用字
    const docFreq = new Map<string, number>();
    const parsedDocs: Array<{ category: QaCategory; tokens: Array<{ tok: string; w: number }> }> = [];

    for (const sample of trainSamples) {
      const tokens = this.extractWeightedNgrams(sample.question);
      parsedDocs.push({ category: sample.category, tokens });
      this.classDocCounts[sample.category] += 1;
      this.totalDocs += 1;

      const uniqueToks = new Set(tokens.map((t) => t.tok));
      for (const tok of uniqueToks) {
        this.vocab.add(tok);
        docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
      }
    }

    for (const [tok, df] of docFreq.entries()) {
      const idf = Math.log((this.totalDocs + 1) / (df + 1)) + 1;
      this.tokenIdf.set(tok, idf);
    }

    for (const doc of parsedDocs) {
      for (const { tok, w } of doc.tokens) {
        const idf = this.tokenIdf.get(tok) ?? 1;
        const weight = w * idf;
        const prev = this.classTokenWeights[doc.category].get(tok) ?? 0;
        this.classTokenWeights[doc.category].set(tok, prev + weight);
        this.classTotalWeight[doc.category] += weight;
      }
    }
  }

  predict(text: string): QaCategory {
    if (this.totalDocs === 0) {
      return "consultation";
    }

    const tokens = this.extractWeightedNgrams(text);
    const vocabSize = Math.max(1, this.vocab.size);
    let bestCat: QaCategory = "consultation";
    let bestLogProb = -Infinity;

    for (const cat of CATEGORIES) {
      const docCount = this.classDocCounts[cat];
      if (docCount === 0) continue;

      let logProb = Math.log(docCount / this.totalDocs);
      const denom = this.classTotalWeight[cat] + this.alpha * vocabSize;

      for (const { tok, w } of tokens) {
        const idf = this.tokenIdf.get(tok) ?? 1;
        const count = this.classTokenWeights[cat].get(tok) ?? 0;
        logProb += w * idf * Math.log((count + this.alpha) / denom);
      }

      if (logProb > bestLogProb) {
        bestLogProb = logProb;
        bestCat = cat;
      }
    }

    return bestCat;
  }

  private extractWeightedNgrams(
    text: string
  ): Array<{ tok: string; w: number }> {
    const cleaned = text
      .replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, "")
      .toLowerCase();
    const grams: Array<{ tok: string; w: number }> = [];

    for (let n = this.ngramMin; n <= this.ngramMax; n += 1) {
      const weight = n >= 2 ? this.bigramWeight : 1.0;
      for (let i = 0; i <= cleaned.length - n; i += 1) {
        const tok = cleaned.slice(i, i + n);
        if (n === 1 && STOP_CHARS.has(tok)) {
          continue;
        }
        grams.push({ tok, w: weight });
      }
    }
    return grams;
  }
}
