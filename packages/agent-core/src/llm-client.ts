export interface LlmChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LlmClient {
  readonly modelName: string;
  chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }>;
}

/**
 * 本地 Ollama 对话客户端（完全离线运行 3B~7B 量化模型）
 */
export class OllamaLlmClient implements LlmClient {
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
    this.modelName = options?.modelName ?? "qwen2.5:3b";
    this.timeoutMs = options?.timeoutMs ?? 60_000;
  }

  async checkHealth(timeoutMs = 2000): Promise<{
    available: boolean;
    models: string[];
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) {
        return {
          available: false,
          models: [],
          error: `HTTP ${response.status}`,
        };
      }
      const data = (await response.json()) as {
        models?: Array<{ name?: string }>;
      };
      const models = (data.models ?? [])
        .map((m) => m.name)
        .filter((n): n is string => Boolean(n));
      return { available: true, models };
    } catch (error) {
      return {
        available: false,
        models: [],
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(this.timeoutMs),
      body: JSON.stringify({
        model: this.modelName,
        messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama 对话请求失败: ${response.status}`);
    }

    const data = (await response.json()) as {
      message?: { content?: string };
      model?: string;
    };

    if (!data.message?.content) {
      throw new Error("Ollama 返回内容为空");
    }

    return {
      content: data.message.content,
      model: data.model ?? this.modelName,
    };
  }
}

/**
 * 带自动降级保护的对话客户端：主模型不可用或异常时无缝回退至本地兜底模板
 */
export class FallbackAwareLlmClient implements LlmClient {
  readonly modelName: string;
  private readonly primary: LlmClient;
  private readonly fallback: LlmClient;

  constructor(primary: LlmClient, fallback: LlmClient) {
    this.primary = primary;
    this.fallback = fallback;
    this.modelName = primary.modelName;
  }

  async chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }> {
    try {
      return await this.primary.chat(messages);
    } catch {
      const fb = await this.fallback.chat(messages);
      return {
        content: fb.content,
        model: `${fb.model} (fallback from ${this.primary.modelName})`,
      };
    }
  }
}

/**
 * 零依赖离线场景感知回答器（综合 Top-K 命中片段并附带服务引导）
 */
export class TemplateFallbackLlmClient implements LlmClient {
  readonly modelName = "offline-template-fallback";

  async chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }> {
    const systemMsg = messages.find((m) => m.role === "system")?.content ?? "";
    const userMsg =
      [...messages].reverse().find((m) => m.role === "user")?.content ?? "";

    const answers = Array.from(
      systemMsg.matchAll(/标准回答：([^\n]+)/g),
      (m) => m[1].trim()
    );
    const uniqueAnswers = Array.from(new Set(answers));

    if (uniqueAnswers.length === 0) {
      return {
        content:
          "您好，已收到您的问题。当前知识库暂未匹配到明确条款，您可以补充订单号或具体场景，或直接点击“投诉转人工”由人工专员为您处理。",
        model: this.modelName,
      };
    }

    const primary = uniqueAnswers[0];
    const secondary = uniqueAnswers[1];

    let intro = "您好，为您查询到以下处理说明：";
    let closing = "如还有其他疑问，欢迎继续告知我。";
    if (systemMsg.includes("当前识别意图：complaint") || /投诉|破损|质量|举报|欺诈/.test(userMsg)) {
      intro = "非常抱歉给您带来不佳体验，我们高度重视您反馈的问题，先为您提供以下处理方案：";
      closing = "如该方案仍未满足您的诉求，您可以直接回复“转人工”或点击下方“👎 没帮助”，我们将立即为您安排专员跟进。";
    } else if (/物流|快递|几天|发货|配送/.test(userMsg)) {
      closing =
        "如需查询具体包裹进度，您也可以提供订单号或点击下方反馈按钮联系人工专员核实。";
    } else if (/退货|退款|换货/.test(userMsg)) {
      closing =
        "办理售后前建议提前核对订单状态与商品完好情况，如有异常可随时转人工处理。";
    }

    const parts = [`${intro}\n1. ${primary}`];
    if (secondary) {
      parts.push(`2. 补充参考：${secondary}`);
    }
    parts.push(closing);

    return {
      content: parts.join("\n"),
      model: this.modelName,
    };
  }
}
