import type { QaCategory } from "@ts-qa/shared";

const EXPLICIT_LABEL_MAP: Record<string, QaCategory> = {
  consultation: "consultation",
  咨询: "consultation",
  普通咨询: "consultation",
  售前咨询: "consultation",
  logistics: "logistics",
  物流: "logistics",
  快递: "logistics",
  配送: "logistics",
  refund: "refund",
  退款: "refund",
  退货: "refund",
  退换货: "refund",
  售后退款: "refund",
  complaint: "complaint",
  投诉: "complaint",
  客诉: "complaint",
  维权: "complaint",
};

/**
 * 将公开语料原始类别或问题正文映射为 4 大标准类别
 */
export function mapToStandardCategory(
  rawCategory: string | undefined,
  questionText: string
): QaCategory {
  if (rawCategory) {
    const normalized = rawCategory.trim().toLowerCase();
    if (normalized in EXPLICIT_LABEL_MAP) {
      return EXPLICIT_LABEL_MAP[normalized];
    }
  }

  const text = questionText.toLowerCase();
  if (/投诉|举报|维权|差评|态度恶劣|欺诈|破损赔偿|一直不处理/.test(text)) {
    return "complaint";
  }
  if (/退货|退款|换货|退钱|七天无理由|仅退款/.test(text)) {
    return "refund";
  }
  if (/物流|快递|发货|配送|揽收|卡住|没收到|单号/.test(text)) {
    return "logistics";
  }
  return "consultation";
}
