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
  private readonly timeoutMs: number;

  constructor(options?: {
    baseUrl?: string;
    modelName?: string;
    timeoutMs?: number;
  }) {
    this.baseUrl = (options?.baseUrl ?? "http://127.0.0.1:11434").replace(
      /\/+$/,
      ""
    );
    this.modelName = options?.modelName ?? "bge-m3";
    this.timeoutMs = options?.timeoutMs ?? 15_000;
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(this.timeoutMs),
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

const EMBED_STOP_CHARS = new Set([
  "的", "了", "吗", "呢", "啊", "吧", "呀", "么", "这", "那",
  "我", "你", "您", "他", "她", "们", "在", "是", "有", "和",
  "与", "及", "还", "能", "会", "要", "都", "就", "也", "很",
]);

/**
 * 业务同义概念词表：将口语化/同义表达映射到统一概念 Token，显著提升复杂同义表达的检索召回率
 */
const SEMANTIC_CONCEPT_GROUPS: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["__C_DAMAGED__", ["破损", "碎了", "摔碎", "摔坏", "压坏", "裂开", "损坏", "漏液", "变形", "掉漆", "发霉", "折断"]],
  ["__C_COMPENSATE__", ["赔付", "赔偿", "补偿", "理赔", "赔钱", "赔款", "赔礼"]],
  ["__C_REFUND__", ["退款", "退钱", "退回账户", "退回", "返还", "差价"]],
  ["__C_RETURN_GOODS__", ["退货", "退回去", "无理由", "寄回", "退回仓", "拒收"]],
  ["__C_EXCHANGE__", ["换货", "换码", "换一个", "调换", "补发", "重发"]],
  ["__C_SHIPPING_FEE__", ["运费", "邮费", "快递费", "运费险", "寄件费"]],
  ["__C_LOGISTICS__", ["物流", "快递", "包裹", "快件", "配送", "运输", "发货", "派送", "送达", "揽收"]],
  ["__C_TRACKING_STUCK__", ["卡住", "停滞", "不动", "没动静", "不更新", "延误", "超时", "疑难件", "催促", "加急"]],
  ["__C_SIGN_ISSUE__", ["签收", "没收到", "未收到", "代签", "丢失", "丢件", "误投", "取件码"]],
  ["__C_INTERCEPT__", ["拦截", "追回", "退回网点", "中途退回"]],
  ["__C_MISSING__", ["少发", "漏发", "缺件", "缺了", "少了", "少了一", "空的", "缺失", "零件", "配件"]],
  ["__C_WRONG_ITEM__", ["发错", "错发", "货不对板", "不是这款", "发成"]],
  ["__C_COUNTERFEIT__", ["假货", "伪劣", "仿品", "劣质", "山寨", "质量差", "不合格", "防伪"]],
  ["__C_INVOICE__", ["发票", "开票", "税号", "抬头", "电子票", "专票", "普票"]],
  ["__C_COUPON__", ["优惠券", "满减", "折扣", "代金券", "红包", "抵扣", "立减"]],
  ["__C_PRICE_PROTECT__", ["价保", "保价", "降价", "买贵"]],
  ["__C_MODIFY_ADDR__", ["改地址", "修改地址", "地址填错", "门牌号", "转寄", "改电话"]],
  ["__C_CUSTOMS__", ["清关", "海关", "保税", "直邮", "跨境", "关税", "实名"]],
  ["__C_SERVICE_BAD__", ["态度差", "敷衍", "不耐烦", "踢皮球", "挂电话", "失联", "投诉客服"]],
  ["__C_SAFETY__", ["过期", "异物", "冒烟", "过敏", "划伤", "隐私泄露", "诈骗", "盗刷"]],
];

/**
 * 可训练 TF-IDF 向量器：从当前训练集学习词表与 IDF 权重，让训练参数与语料直接决定检索结果
 */
export class TrainedTfidfEmbedder implements Embedder {
  modelName: string;
  private ngramMax: number;
  private readonly maxFeatures: number;
  private vocabIndex = new Map<string, number>();
  private idfWeights: number[] = [];

  constructor(options?: { ngramMax?: number; maxFeatures?: number }) {
    this.ngramMax = options?.ngramMax ?? 2;
    this.maxFeatures = options?.maxFeatures ?? 1024;
    this.modelName = `trained-tfidf(n=1..${this.ngramMax})`;
  }

  fit(texts: readonly string[], ngramMax?: number): void {
    if (ngramMax !== undefined) {
      this.ngramMax = ngramMax;
      this.modelName = `trained-tfidf(n=1..${this.ngramMax})`;
    }
    this.vocabIndex.clear();
    this.idfWeights = [];

    const docFreq = new Map<string, number>();
    for (const text of texts) {
      const uniqueToks = new Set(this.tokenize(text).map((t) => t.tok));
      for (const tok of uniqueToks) {
        docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
      }
    }

    const totalDocs = Math.max(1, texts.length);
    const conceptEntries: Array<[string, number]> = [];
    const ngramEntries: Array<[string, number]> = [];
    for (const entry of docFreq.entries()) {
      if (entry[0].startsWith("__concept_")) {
        conceptEntries.push(entry);
      } else {
        ngramEntries.push(entry);
      }
    }
    ngramEntries.sort((a, b) => b[1] - a[1]);
    const remainingSlots = Math.max(0, this.maxFeatures - conceptEntries.length);
    const selectedEntries = [
      ...conceptEntries,
      ...ngramEntries.slice(0, remainingSlots),
    ];

    let idx = 0;
    for (const [tok, df] of selectedEntries) {
      this.vocabIndex.set(tok, idx);
      this.idfWeights.push(Math.log((totalDocs + 1) / (df + 1)) + 1);
      idx += 1;
    }
  }

  async embed(text: string): Promise<number[]> {
    const dims = this.vocabIndex.size;
    if (dims === 0) {
      return new LocalNgramEmbedder().embed(text);
    }

    const vec = new Array<number>(dims).fill(0);
    for (const { tok, w } of this.tokenize(text)) {
      const idx = this.vocabIndex.get(tok);
      if (idx !== undefined) {
        vec[idx] += w * this.idfWeights[idx];
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

  private tokenize(text: string): Array<{ tok: string; w: number }> {
    const rawLower = text.toLowerCase();
    const cleaned = text
      .replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, "")
      .toLowerCase();
    const out: Array<{ tok: string; w: number }> = [];
    for (let n = 1; n <= this.ngramMax; n += 1) {
      const weight = n >= 2 ? 2.5 : 1.0;
      for (let i = 0; i <= cleaned.length - n; i += 1) {
        const tok = cleaned.slice(i, i + n);
        if (n === 1 && EMBED_STOP_CHARS.has(tok)) continue;
        out.push({ tok, w: weight });
      }
    }
    for (const [concept, keywords] of SEMANTIC_CONCEPT_GROUPS) {
      if (keywords.some((kw) => rawLower.includes(kw) || cleaned.includes(kw))) {
        out.push({ tok: concept, w: 5.0 });
      }
    }
    return out;
  }
}

/**
 * 混合语义向量器：将语义向量（如 Ollama Embedding）与同义概念增强 TF-IDF 拼接融合，
 * 兼顾长句同义理解与专有关键词精确匹配；语义服务不可用时自动降级为 TF-IDF。
 */
export class HybridSemanticEmbedder implements Embedder {
  readonly modelName: string;
  private readonly semanticEmbedder: Embedder;
  private readonly lexicalEmbedder: TrainedTfidfEmbedder;
  private readonly semanticWeight: number;

  constructor(options: {
    semanticEmbedder: Embedder;
    lexicalEmbedder: TrainedTfidfEmbedder;
    semanticWeight?: number;
  }) {
    this.semanticEmbedder = options.semanticEmbedder;
    this.lexicalEmbedder = options.lexicalEmbedder;
    this.semanticWeight = options.semanticWeight ?? 0.65;
    this.modelName = `hybrid(${this.semanticEmbedder.modelName}+${this.lexicalEmbedder.modelName})`;
  }

  async embed(text: string): Promise<number[]> {
    const lexVec = await this.lexicalEmbedder.embed(text);
    try {
      const semVec = await this.semanticEmbedder.embed(text);
      const sw = Math.sqrt(this.semanticWeight);
      const lw = Math.sqrt(1 - this.semanticWeight);
      const combined = new Array<number>(semVec.length + lexVec.length);
      for (let i = 0; i < semVec.length; i += 1) {
        combined[i] = semVec[i] * sw;
      }
      for (let i = 0; i < lexVec.length; i += 1) {
        combined[semVec.length + i] = lexVec[i] * lw;
      }
      const norm = Math.sqrt(combined.reduce((acc, v) => acc + v * v, 0));
      if (norm > 0) {
        for (let i = 0; i < combined.length; i += 1) {
          combined[i] /= norm;
        }
      }
      return combined;
    } catch {
      return lexVec;
    }
  }
}
