export interface SanitizeResult {
  cleanedText: string;
  sanitized: boolean;
}

const PHONE_REGEX = /(?<!\d)1[3-9]\d{9}(?!\d)/g;
const ID_CARD_REGEX = /(?<!\d)\d{17}[\dXx](?!\d)/g;
const EMAIL_REGEX = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
const BANK_CARD_REGEX = /(?<!\d)\d{16,19}(?!\d)/g;

/**
 * 对中文客服文本进行空白规范化与敏感信息脱敏
 */
export function sanitizeText(input: string): SanitizeResult {
  let text = input.replace(/\r\n/g, "\n").replace(/[ \t]+/g, " ").trim();
  let sanitized = false;

  const rules: Array<[RegExp, string]> = [
    [ID_CARD_REGEX, "[ID_CARD]"],
    [BANK_CARD_REGEX, "[BANK_CARD]"],
    [PHONE_REGEX, "[PHONE]"],
    [EMAIL_REGEX, "[EMAIL]"],
  ];

  for (const [pattern, replacement] of rules) {
    const next = text.replace(pattern, replacement);
    if (next !== text) {
      sanitized = true;
      text = next;
    }
  }

  return { cleanedText: text, sanitized };
}
