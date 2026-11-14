import { createHash, randomUUID } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { basename, extname } from "node:path";
import type { DocumentSourceType, RawDocument } from "@ts-qa/shared";
import type { SqliteStore } from "../storage/sqlite-store.ts";

/**
 * 解析后的原始语料单条记录（兼容多种公开中文客服/投诉数据集字段名）
 */
export interface RawCorpusRecord {
  question: string;
  answer: string;
  rawCategory?: string;
  intentTag?: string;
}

export interface CorpusImportResult {
  document: RawDocument;
  records: RawCorpusRecord[];
  deduplicated: boolean;
}

/**
 * 解析 JSONL 文本为标准原始语料记录
 */
export function parseJsonlCorpus(content: string): RawCorpusRecord[] {
  const records: RawCorpusRecord[] = [];
  const lines = content.split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const item = JSON.parse(trimmed) as Record<string, unknown>;
    const question = pickString(item, ["question", "query", "prompt", "input", "customer_utterance"]);
    const answer = pickString(item, ["answer", "reply", "response", "output", "agent_utterance"]);
    const rawCategory = pickString(item, ["category", "label", "type", "complaint_type"]);
    const intentTag = pickString(item, ["intent", "intentTag", "sub_category"]);

    if (question && answer) {
      records.push({
        question,
        answer,
        rawCategory: rawCategory || undefined,
        intentTag: intentTag || undefined,
      });
    }
  }

  return records;
}

/**
 * 解析带表头的 CSV 文本（支持双引号包裹及内部逗号）
 */
export function parseCsvCorpus(content: string): RawCorpusRecord[] {
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];

  const headers = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
  const records: RawCorpusRecord[] = [];

  for (let i = 1; i < lines.length; i += 1) {
    const cols = splitCsvLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = (cols[idx] ?? "").trim();
    });

    const question = pickString(row, ["question", "query", "prompt", "input"]);
    const answer = pickString(row, ["answer", "reply", "response", "output"]);
    const rawCategory = pickString(row, ["category", "label", "type"]);
    const intentTag = pickString(row, ["intent", "intent_tag"]);

    if (question && answer) {
      records.push({
        question,
        answer,
        rawCategory: rawCategory || undefined,
        intentTag: intentTag || undefined,
      });
    }
  }

  return records;
}

/**
 * 导入本地语料文件并写入 SQLite，支持按 SHA-256 校验和去重
 */
export function importCorpusFile(
  filePath: string,
  store: SqliteStore,
  options?: { title?: string }
): CorpusImportResult {
  const rawBuffer = readFileSync(filePath);
  const checksum = createHash("sha256").update(rawBuffer).digest("hex");
  const content = rawBuffer.toString("utf-8");
  const ext = extname(filePath).toLowerCase();

  const records =
    ext === ".csv" ? parseCsvCorpus(content) : parseJsonlCorpus(content);

  const existing = store.getRawDocumentByChecksum(checksum);
  if (existing) {
    return {
      document: existing,
      records,
      deduplicated: true,
    };
  }

  const now = new Date().toISOString();
  const stats = statSync(filePath);
  const sourceType: DocumentSourceType =
    ext === ".csv" ? "csv" : "chat_log";

  const doc: RawDocument = {
    id: randomUUID(),
    title: options?.title ?? basename(filePath),
    sourceType,
    filePath,
    checksum,
    sizeBytes: stats.size,
    status: "pending",
    createdAt: now,
    updatedAt: now,
  };

  store.upsertRawDocument(doc);

  return {
    document: doc,
    records,
    deduplicated: false,
  };
}

function pickString(
  obj: Record<string, unknown>,
  keys: string[]
): string {
  for (const key of keys) {
    const val = obj[key];
    if (typeof val === "string" && val.trim().length > 0) {
      return val.trim();
    }
  }
  return "";
}

function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}
