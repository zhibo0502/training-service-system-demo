/**
 * 公开中文客服与投诉语料转换脚本
 *
 * 用法：
 *   node ./scripts/prepare-public-corpus.ts <输入文件路径(.json/.jsonl/.csv)> <输出JSONL路径>
 *
 * 支持字段自动映射：
 * - 提问字段：question / query / prompt / input / customer_utterance / content
 * - 回答字段：answer / reply / response / output / agent_utterance / solution
 * - 类别字段：category / label / type / complaint_type（若缺失则按关键词自动归类）
 */
import { readFileSync, writeFileSync } from "node:fs";
import { extname, resolve } from "node:path";
import { mapToStandardCategory, sanitizeText } from "../packages/cleaner/src/index.ts";
import { parseCsvCorpus, parseJsonlCorpus } from "../packages/collector/src/index.ts";

const [, , inputArg, outputArg] = process.argv;

if (!inputArg || !outputArg) {
  console.error(
    "用法: node ./scripts/prepare-public-corpus.ts <input_file> <output_jsonl>"
  );
  process.exit(1);
}

const inputPath = resolve(inputArg);
const outputPath = resolve(outputArg);
const raw = readFileSync(inputPath, "utf-8");
const ext = extname(inputPath).toLowerCase();

let records = [];
if (ext === ".csv") {
  records = parseCsvCorpus(raw);
} else if (ext === ".json") {
  const parsed = JSON.parse(raw);
  const arr = Array.isArray(parsed) ? parsed : [];
  records = parseJsonlCorpus(arr.map((item) => JSON.stringify(item)).join("\n"));
} else {
  records = parseJsonlCorpus(raw);
}

const outputLines: string[] = [];
for (const item of records) {
  const q = sanitizeText(item.question).cleanedText;
  const a = sanitizeText(item.answer).cleanedText;
  if (q.length < 2 || a.length < 2) continue;
  const category = mapToStandardCategory(item.rawCategory, q);
  outputLines.push(JSON.stringify({ question: q, answer: a, category }));
}

writeFileSync(outputPath, `${outputLines.join("\n")}\n`, "utf-8");
console.log(`转换完成：共输出 ${outputLines.length} 条标准语料至 ${outputPath}`);
