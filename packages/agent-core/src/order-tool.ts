import type { QaCategory } from "@ts-qa/shared";

export interface OrderInfo {
  orderId: string;
  status: string;
  carrier: string;
  trackingNo: string;
  latestEvent: string;
  refundEligible: boolean;
}

const PRESET_ORDERS: Record<string, OrderInfo> = {
  "ORD-1001": {
    orderId: "ORD-1001",
    status: "运输中",
    carrier: "顺丰速运",
    trackingNo: "SF1029384756",
    latestEvent: "已到达杭州集散中心，预计明日派送",
    refundEligible: true,
  },
  "ORD-1002": {
    orderId: "ORD-1002",
    status: "已签收",
    carrier: "中通快递",
    trackingNo: "ZTO9988776655",
    latestEvent: "本人已签收（7天无理由保障期内）",
    refundEligible: true,
  },
  "ORD-1003": {
    orderId: "ORD-1003",
    status: "退款审核中",
    carrier: "京东物流",
    trackingNo: "JD0011223344",
    latestEvent: "退货包裹已揽收，预计1-2个工作日原路退回",
    refundEligible: true,
  },
};

/**
 * 从客户文本中提取订单号（支持 ORD-xxxx 或 10-18 位纯数字单号）
 */
export function extractOrderId(text: string): string | null {
  const ordMatch = text.match(/\b(ORD-\d{4,12})\b/i);
  if (ordMatch) {
    return ordMatch[1].toUpperCase();
  }
  const numMatch = text.match(/(?<!\d)(\d{10,18})(?!\d)/);
  if (numMatch) {
    return numMatch[1];
  }
  return null;
}

/**
 * 查询本地订单与物流工具（模拟真实业务系统接口）
 */
export function lookupOrder(orderId: string): OrderInfo {
  const normalized = orderId.toUpperCase();
  if (PRESET_ORDERS[normalized]) {
    return PRESET_ORDERS[normalized];
  }
  const lastDigit = Number(normalized.slice(-1)) || 0;
  const statuses = ["已发货", "运输中", "派送中", "已签收"];
  const carriers = ["顺丰速运", "中通快递", "圆通速递", "京东物流"];
  const status = statuses[lastDigit % statuses.length];
  const carrier = carriers[lastDigit % carriers.length];
  return {
    orderId: normalized,
    status,
    carrier,
    trackingNo: `TRK${normalized.replace(/\D/g, "").slice(-8).padStart(8, "0")}`,
    latestEvent: `当前订单状态：${status}（${carrier}正常履约中）`,
    refundEligible: true,
  };
}

const SUB_INTENT_RULES: Array<{ subIntent: string; pattern: RegExp }> = [
  { subIntent: "modify_address", pattern: /改地址|修改收货|换地址|地址写错/ },
  { subIntent: "invoice_request", pattern: /发票|开票|税号|报销/ },
  { subIntent: "price_protection", pattern: /保价|降价|差价|买贵/ },
  { subIntent: "missing_item", pattern: /少发|漏发|少件|没收到赠品/ },
  { subIntent: "damaged_compensation", pattern: /摔碎|破损|压扁|坏了|碎了|赔付|赔偿/ },
  { subIntent: "courier_delay", pattern: /几天|催发货|没动静|卡住|延迟|物流|快递|到哪|配送|派送/ },
  { subIntent: "return_shipping", pattern: /退货运费|运费险|谁出运费|寄回/ },
  { subIntent: "refund_progress", pattern: /退款进度|到账|退钱|仅退款|退货/ },
  { subIntent: "coupon_rule", pattern: /优惠券|礼券|满减|积分|折扣/ },
  { subIntent: "warranty", pattern: /质保|联保|保修|维修|售后/ },
];

/**
 * 在 4 个一级意图基础上识别细分二级业务意图
 */
export function detectSubIntent(
  question: string,
  category?: QaCategory
): string {
  for (const rule of SUB_INTENT_RULES) {
    if (rule.pattern.test(question)) {
      return rule.subIntent;
    }
  }
  return `general_${category ?? "consultation"}`;
}
