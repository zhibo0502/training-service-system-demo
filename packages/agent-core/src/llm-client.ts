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
 * 零依赖离线模板回答器（未启动 Ollama 时基于命中知识库片段直接组装回复）
 */
export class TemplateFallbackLlmClient implements LlmClient {
  readonly modelName = "offline-template-fallback";

  async chat(messages: readonly LlmChatMessage[]): Promise<{
    content: string;
    model: string;
  }> {
    const systemMsg = messages.find((m) => m.role === "system")?.content ?? "";
    const answerMatch = systemMsg.match(/标准回答：([^\n]+)/);
    if (answerMatch?.[1]) {
      return {
        content: `您好，为您查询到相关处理说明：${answerMatch[1].trim()}`,
        model: this.modelName,
      };
    }

    return {
      content:
        "您好，已收到您的问题。当前离线模型未检索到足够匹配的参考条目，如需进一步协助可直接点击转人工。",
      model: this.modelName,
    };
  }
}
