import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export interface VectorRecord {
  id: string;
  documentId: string;
  content: string;
  category?: string;
  vector: number[];
}

export interface VectorSearchHit {
  record: VectorRecord;
  score: number;
}

/**
 * 基于本地 JSON 文件的单机轻量级向量索引库（零 Docker 依赖）
 */
export class LocalVectorStore {
  private readonly filePath?: string;
  private records = new Map<string, VectorRecord>();

  constructor(filePath?: string) {
    this.filePath = filePath;
    if (filePath && existsSync(filePath)) {
      const raw = readFileSync(filePath, "utf-8");
      const parsed = JSON.parse(raw) as VectorRecord[];
      for (const item of parsed) {
        this.records.set(item.id, item);
      }
    }
  }

  upsert(items: readonly VectorRecord[]): void {
    for (const item of items) {
      this.records.set(item.id, item);
    }
    this.persist();
  }

  replaceAll(items: readonly VectorRecord[]): void {
    this.records.clear();
    for (const item of items) {
      this.records.set(item.id, item);
    }
    this.persist();
  }

  size(): number {
    return this.records.size;
  }

  listRecords(): VectorRecord[] {
    return Array.from(this.records.values());
  }

  search(queryVector: readonly number[], topK = 3): VectorSearchHit[] {
    const scored: VectorSearchHit[] = [];

    for (const record of this.records.values()) {
      const score = cosineSimilarity(queryVector, record.vector);
      scored.push({ record, score });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  private persist(): void {
    if (!this.filePath) return;
    mkdirSync(dirname(this.filePath), { recursive: true });
    const data = Array.from(this.records.values());
    writeFileSync(this.filePath, JSON.stringify(data, null, 2), "utf-8");
  }
}

export function cosineSimilarity(
  a: readonly number[],
  b: readonly number[]
): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < len; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
