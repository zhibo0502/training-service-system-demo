export interface GuardrailResult {
  safeText: string;
  triggered: boolean;
  reason?: string;
}

const FORBIDDEN_PROMISE_REGEX =
  /(无条件赔十倍|全额免单再赔|私下转账|加我微信|加微信|私聊返现|个人支付宝)/g;
const PHONE_REGEX = /(?<!\d)1[3-9]\d{9}(?!\d)/g;
const ID_CARD_REGEX = /(?<!\d)\d{17}[\dXx](?!\d)/g;

/**
 * 客服回复合规护栏：拦截越权赔付承诺、私下交易引导与敏感信息泄露
 */
export function applyComplianceGuardrail(replyText: string): GuardrailResult {
  let safeText = replyText;
  const reasons: string[] = [];

  if (FORBIDDEN_PROMISE_REGEX.test(safeText)) {
    reasons.push("unauthorized_promise_or_offplatform");
    safeText = safeText.replace(
      FORBIDDEN_PROMISE_REGEX,
      "按平台官方售后规则处理"
    );
  }

  if (PHONE_REGEX.test(safeText)) {
    reasons.push("pii_phone_masked");
    safeText = safeText.replace(PHONE_REGEX, "[PHONE]");
  }

  if (ID_CARD_REGEX.test(safeText)) {
    reasons.push("pii_id_masked");
    safeText = safeText.replace(ID_CARD_REGEX, "[ID_CARD]");
  }

  return {
    safeText,
    triggered: reasons.length > 0,
    reason: reasons.length > 0 ? reasons.join(",") : undefined,
  };
}
