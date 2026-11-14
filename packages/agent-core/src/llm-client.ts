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

  constructor(options?: { baseUrl?: string; modelName?: string }) {
    this.baseUrl = options?.baseUrl ?? "http://127.0.0.1:11434";
    this.modelName = options?.modelName ?? "qwen2.5:3b";
  }

  async chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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

    let closing = "如还有其他疑问，欢迎继续告知我。";
    if (/物流|快递|几天|发货|配送/.test(userMsg)) {
      closing =
        "如需查询具体包裹进度，您也可以提供订单号或点击下方反馈按钮联系人工专员核实。";
    } else if (/退货|退款|换货/.test(userMsg)) {
      closing =
        "办理售后前建议提前核对订单状态与商品完好情况，如有异常可随时转人工处理。";
    }

    const parts = [`您好，为您查询到以下处理说明：\n1. ${primary}`];
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
