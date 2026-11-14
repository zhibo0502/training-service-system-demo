/**
 * 文本向量化器接口
 */
export interface Embedder {
  readonly modelName: string;
  embed(text: string): Promise<number[]>;
}

/**
 * 本地 Ollama Embedding 客户端（完全离线）
 */
export class OllamaEmbedder implements Embedder {
  readonly modelName: string;
  private readonly baseUrl: string;

  constructor(options?: { baseUrl?: string; modelName?: string }) {
    this.baseUrl = options?.baseUrl ?? "http://127.0.0.1:11434";
    this.modelName = options?.modelName ?? "nomic-embed-text";
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.modelName,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding 请求失败: ${response.status}`);
    }

    const data = (await response.json()) as { embedding?: number[] };
    if (!Array.isArray(data.embedding)) {
      throw new Error("Ollama embedding 返回格式异常");
    }
    return data.embedding;
  }
}

/**
 * 零依赖本地中文字符 N-gram 哈希向量器（无 Ollama 时的确定性离线兜底）
 */
export class LocalNgramEmbedder implements Embedder {
  readonly modelName = "local-ngram-hash-256";
  private readonly dims = 256;

  async embed(text: string): Promise<number[]> {
    const vec = new Array<number>(this.dims).fill(0);
    const normalized = text.replace(/\s+/g, "").toLowerCase();

    for (let i = 0; i < normalized.length; i += 1) {
      const uni = normalized.charCodeAt(i);
      vec[uni % this.dims] += 1;

      if (i + 1 < normalized.length) {
        const bi = (uni * 31 + normalized.charCodeAt(i + 1)) >>> 0;
        vec[bi % this.dims] += 1.5;
      }
    }

    const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < vec.length; i += 1) {
        vec[i] /= norm;
      }
    }
    return vec;
  }
}
