import type { StandardCategory } from "@ts-qa/shared";
import { createSeededRng } from "./prng.ts";

export interface SyntheticQaRecord {
  id: string;
  baseKnowledgeId: string;
  sourceType: "base" | "synthetic";
  rawCategory: StandardCategory;
  intentTag: string;
  question: string;
  answer: string;
  variantRule?: string;
}

export interface SynthDatasetOptions {
  baseCount?: number;
  targetCount?: number;
  seed?: number;
}

export interface SynthDatasetResult {
  baseRecords: SyntheticQaRecord[];
  allRecords: SyntheticQaRecord[];
  stats: {
    baseCount: number;
    totalCount: number;
    syntheticCount: number;
    categoryCounts: Record<StandardCategory, number>;
    durationMs: number;
  };
}

interface ScenarioBlueprint {
  category: StandardCategory;
  intentTag: string;
  subjects: readonly string[];
  conditions: readonly string[];
  actions: readonly string[];
  questionTemplates: readonly string[];
  answerTemplate: string;
}

const SCENARIO_BLUEPRINTS: readonly ScenarioBlueprint[] = [
  // ================= consultation (7 个子场景) =================
  {
    category: "consultation",
    intentTag: "promo_coupon_usage",
    subjects: [
      "数码配件", "美妆护肤品", "应季服饰", "家用小电器",
      "母婴用品", "运动鞋服", "进口零食", "图书文创",
    ],
    conditions: [
      "满减活动期间", "新人首单", "会员日专享", "叠加店铺券时",
      "跨店满减结算时", "秒杀特价商品", "预售付尾款时", "使用积分兑换后",
    ],
    actions: ["使用优惠券", "享受折扣", "叠加满减", "领券下单"],
    questionTemplates: [
      "购买{subject}在{condition}可以{action}吗？",
      "请问{condition}买{subject}支持{action}吗？",
      "{subject}如果处于{condition}，还能{action}不？",
      "我想问下{subject}{condition}能不能{action}？",
      "下单{subject}遇到{condition}怎么{action}？",
      "关于{subject}，在{condition}时支持{action}吗？",
    ],
    answerTemplate:
      "关于{subject}在{condition}的优惠说明：普通正价商品支持按页面规则{action}，秒杀或标注“不参与满减”的特价商品以结算页实际可用优惠为准。",
  },
  {
    category: "consultation",
    intentTag: "invoice_request",
    subjects: [
      "电子普通发票", "增值税专用发票", "企业抬头发票", "个人抬头发票",
      "办公耗材订单", "企业采购订单", "数码主机订单", "礼品卡订单",
    ],
    conditions: [
      "下单前忘记备注", "订单已确认收货", "部分商品已退款", "需要修改税号",
      "开票超过30天", "合并多笔订单", "使用钱包余额支付", "跨月补开",
    ],
    actions: ["申请开票", "重开发票", "下载PDF发票", "补开电子发票"],
    questionTemplates: [
      "{subject}如果{condition}，还能{action}吗？",
      "请问{condition}的情况下，{subject}怎么{action}？",
      "我的{subject}因为{condition}，现在可以{action}吗？",
      "咨询一下，{subject}{condition}支持{action}不？",
      "遇到{condition}时，{subject}要如何{action}？",
      "{subject}在{condition}后多久能完成{action}？",
    ],
    answerTemplate:
      "{subject}在{condition}的情况下，您可在订单详情页点击“申请开票/换开”，填写抬头与税号后系统将在 1-3 个工作日内完成{action}并推送至预留邮箱。",
  },
  {
    category: "consultation",
    intentTag: "product_spec_warranty",
    subjects: [
      "蓝牙耳机", "智能手表", "扫地机器人", "空气炸锅",
      "电动牙刷", "便携显示器", "机械键盘", "车载吸尘器",
    ],
    conditions: [
      "日常家用场景", "出差便携使用", "官方质保期内", "非人为损坏时",
      "赠送给亲友后", "连接苹果手机", "连接安卓设备", "海外电压环境下",
    ],
    actions: ["享受联保", "查询保修期", "确认参数规格", "申请延保"],
    questionTemplates: [
      "这款{subject}在{condition}下支持{action}吗？",
      "请问{subject}{condition}怎么{action}？",
      "买{subject}后如果遇到{condition}，能{action}吗？",
      "想了解下{subject}在{condition}时如何{action}？",
      "{subject}针对{condition}有{action}服务吗？",
      "关于{subject}，{condition}的时候可以{action}不？",
    ],
    answerTemplate:
      "{subject}提供官方正品保障与标准质保服务，在{condition}下您可凭订单编号联系客服或前往商品详情页参数栏{action}。",
  },
  {
    category: "consultation",
    intentTag: "payment_methods",
    subjects: [
      "大额家电订单", "预售定金订单", "跨境直邮订单", "拼团订单",
      "企业批量订单", "虚拟充值订单", "定制商品订单", "组合套餐订单",
    ],
    conditions: [
      "银行卡限额时", "想用花呗分期", "使用微信零钱", "切换信用卡支付",
      "使用礼品卡抵扣", "定金支付阶段", "尾款合并支付时", "帮朋友代付",
    ],
    actions: ["完成付款", "免息分期", "更改支付方式", "合并结算"],
    questionTemplates: [
      "{subject}在{condition}支持{action}吗？",
      "请问{subject}遇到{condition}怎么{action}？",
      "下单{subject}时如果{condition}，能{action}不？",
      "想确认下{subject}在{condition}下可以{action}吗？",
      "我的{subject}正处于{condition}，如何{action}？",
      "{subject}{condition}的话，支持在线{action}吗？",
    ],
    answerTemplate:
      "{subject}支持主流在线支付渠道；遇到{condition}时，可在收银台选择对应支付工具或分期期数完成{action}，具体免息标识以收银台展示为准。",
  },
  {
    category: "consultation",
    intentTag: "stock_and_presale",
    subjects: [
      "联名限定款", "新款手机", "冬季羽绒服", "季节限定礼盒",
      "定制刻字商品", "进口保健品", "热门护肤套装", "儿童安全座椅",
    ],
    conditions: [
      "显示库存紧张", "处于预售阶段", "尺码暂时缺货", "颜色断货",
      "仓库调拨中", "付完定金后", "活动即将结束", "多件打包购买",
    ],
    actions: ["预约补货", "锁定库存", "查询到货时间", "修改收货规格"],
    questionTemplates: [
      "{subject}目前{condition}，能{action}吗？",
      "请问{subject}如果{condition}，什么时候可以{action}？",
      "看中了{subject}但{condition}，怎么{action}？",
      "想问下{subject}{condition}的话，支持{action}不？",
      "购买{subject}遇到{condition}，有没有办法{action}？",
      "{subject}在{condition}状态下能提前{action}吗？",
    ],
    answerTemplate:
      "{subject}在{condition}时，您可点击商品页“到货提醒”或关注预售页面标注的预计发货节点；付款成功后系统将自动为您{action}。",
  },
  {
    category: "consultation",
    intentTag: "account_membership",
    subjects: [
      "Plus年度会员", "店铺VIP等级", "积分兑换权益", "生日专属礼包",
      "运费券包", "成长值累计", "家庭共享账号", "企业采购账号",
    ],
    conditions: [
      "续费前", "更换绑定手机号后", "跨平台登录时", "发生退货退款后",
      "接近月底到期时", "升级会员等级后", "首次开通时", "未收到短信通知",
    ],
    actions: ["查询有效期", "恢复权益", "领取对应福利", "查看规则"],
    questionTemplates: [
      "{subject}在{condition}怎么{action}？",
      "请问{subject}遇到{condition}会影响{action}吗？",
      "我的{subject}因为{condition}，现在怎么{action}？",
      "关于{subject}，在{condition}时能{action}不？",
      "想咨询{subject}{condition}后如何{action}？",
      "{subject}如果{condition}，还可以{action}吗？",
    ],
    answerTemplate:
      "关于{subject}，在{condition}时系统会自动同步最新权益状态；您可在“我的-会员中心”核对明细并按指引{action}。",
  },
  {
    category: "consultation",
    intentTag: "order_modification",
    subjects: [
      "普通现货订单", "生鲜冷链订单", "定制印字订单", "多件合并订单",
      "活动赠品订单", "保税仓直发订单", "同城急送订单", "大件家电订单",
    ],
    conditions: [
      "刚付款5分钟内", "还未生成运单号", "选错了商品颜色", "填错了收货电话",
      "漏选了随单赠品", "想合并两个订单", "需要备注配送时间", "仓库尚未拣货",
    ],
    actions: ["修改地址", "更换规格", "补充备注", "取消重拍"],
    questionTemplates: [
      "{subject}{condition}，还能{action}吗？",
      "请问{subject}在{condition}时怎么{action}？",
      "刚买了{subject}但{condition}，现在可以{action}不？",
      "我的{subject}目前{condition}，支持自助{action}吗？",
      "遇到{subject}{condition}的情况，如何快速{action}？",
      "{subject}如果{condition}，联系客服能帮忙{action}吗？",
    ],
    answerTemplate:
      "若{subject}仍处于{condition}且未出库，您可在订单详情页直接点击“修改信息”完成{action}；若已出库则建议联系快递拦截或拒收后重拍。",
  },
  {
    category: "consultation",
    intentTag: "presale_coupon_boundary",
    subjects: [
      "新人首单立减券", "跨店满减活动", "会员积分抵扣", "店铺大额优惠券",
      "限时秒杀名额", "买赠活动赠品", "保价服务规则", "平台红包门槛",
    ],
    conditions: [
      "还没下单想先了解", "想凑单达到满减门槛", "不确定能否和会员折上折",
      "不知道优惠券有效期到几号", "想了解使用规则与限制", "想知道哪些商品不参与",
      "对比不同支付优惠力度", "下单前核对活动截止时间",
    ],
    actions: ["确认适用规则", "查询领取入口", "核对优惠明细", "了解活动说明"],
    questionTemplates: [
      "请问{subject}在{condition}时怎么{action}？",
      "还没付款，想先咨询下{subject}{condition}怎么{action}？",
      "关于{subject}，如果{condition}的话去哪里{action}？",
      "买之前想确认下，{subject}{condition}可以{action}吗？",
      "{subject}的规则有点看不懂，{condition}怎么{action}？",
      "想问一下{subject}，在{condition}时如何{action}？",
    ],
    answerTemplate:
      "关于{subject}的售前规则：在{condition}时您可直接在活动页或商品详情页“优惠说明”栏{action}，结算页会自动匹配最优可用优惠。",
  },
  {
    category: "consultation",
    intentTag: "account_security_service",
    subjects: [
      "绑定手机号", "账户登录密码", "实名认证信息", "人工客服工作时间",
      "企业采购账号", "会员积分商城", "消息提醒设置", "默认收货地址簿",
    ],
    conditions: [
      "原手机号已停用", "想确认几点到几点在线", "周末或节假日咨询",
      "更换新手机登录", "忘记原账号密码", "需要升级企业认证",
      "想关闭营销短信", "查询账户成长值明细",
    ],
    actions: ["完成更换绑定", "联系人工值班", "查看操作入口", "核验账户安全"],
    questionTemplates: [
      "请问{subject}如果{condition}，怎么{action}？",
      "想咨询一下，{subject}{condition}在哪里{action}？",
      "我的{subject}因为{condition}，现在怎么{action}？",
      "关于{subject}，{condition}时如何{action}？",
      "请问平台{subject}遇到{condition}怎么{action}？",
      "麻烦问下{subject}{condition}的话能{action}吗？",
    ],
    answerTemplate:
      "关于{subject}在{condition}的咨询：您可前往“账户安全中心/帮助中心”按指引{action}；人工客服每日 9:00-22:00 在线，其余时段可提交留言工单。",
  },
  {
    category: "consultation",
    intentTag: "invoice_boundary_query",
    subjects: [
      "电子普通发票", "增值税专用发票", "企业报销发票", "个人抬头电子票",
      "纸质发票寄送", "多笔订单合并开票", "定金尾款发票", "礼品卡消费发票",
    ],
    conditions: [
      "下单前想确认开票类型", "需要把个人抬头改成公司抬头", "不确定税率是多少",
      "想确认多久能发送到邮箱", "需要把普票换成专票", "开票信息填错了想重开",
      "想先了解开票入口在哪", "需要开具明细清单",
    ],
    actions: ["确认开票流程", "修改抬头税号", "申请换开重开", "查询电子票进度"],
    questionTemplates: [
      "请问{subject}{condition}，怎么{action}？",
      "下单前想问下，{subject}如果{condition}如何{action}？",
      "我的{subject}因为{condition}，在哪里{action}？",
      "咨询开票问题：{subject}{condition}可以{action}吗？",
      "关于{subject}，遇到{condition}要怎么{action}？",
      "想了解下{subject}在{condition}时怎样{action}？",
    ],
    answerTemplate:
      "{subject}在{condition}时，可在“我的订单-发票服务”自助提交抬头与税号完成{action}，电子发票开具后将发送至您的预留邮箱。",
  },

  // ================= logistics (7 个子场景) =================
  {
    category: "logistics",
    intentTag: "shipping_speed",
    subjects: [
      "现货普通包裹", "生鲜冷链包裹", "同城闪送订单", "预售尾款订单",
      "保税区发货商品", "大件家具包裹", "定制商品包裹", "节假日促销订单",
    ],
    conditions: [
      "今天下午4点前下单", "发往偏远地区", "周末下单", "大促爆单期间",
      "跨省远距离运输", "发往港澳台地区", "选择标准快递", "选择加急空运",
    ],
    actions: ["安排发货", "送达收货地", "生成物流单号", "发出包裹"],
    questionTemplates: [
      "{subject}在{condition}大概多久能{action}？",
      "请问{subject}如果{condition}，几天可以{action}？",
      "买了{subject}且{condition}，什么时候能{action}？",
      "{subject}{condition}的话，一般几天{action}？",
      "想问下{subject}在{condition}下最快多久{action}？",
      "下单{subject}遇到{condition}，预计啥时候{action}？",
    ],
    answerTemplate:
      "{subject}在{condition}时通常会在 24-48 小时内完成出库并{action}，省内一般 1-2 天送达，跨省约 2-4 天，大促或偏远地区会顺延 1-2 天。",
  },
  {
    category: "logistics",
    intentTag: "tracking_stuck",
    subjects: [
      "顺丰快递单", "中通包裹", "圆通包裹", "京东物流件",
      "邮政EMS件", "韵达包裹", "跨境清关件", "大件专线件",
    ],
    conditions: [
      "停在中转站超过48小时", "显示已揽收但没动静", "到达本地网点两天未派送",
      "物流信息一直空白", "清关状态停滞3天", "分拣中心重复扫描",
      "天气原因延误", "显示疑难件",
    ],
    actions: ["催促派送", "核实物流进度", "联系快递网点", "排查卡顿原因"],
    questionTemplates: [
      "我的{subject}{condition}了，能帮忙{action}吗？",
      "请问{subject}出现{condition}怎么办，可以{action}吗？",
      "{subject}目前{condition}，麻烦帮我{action}一下。",
      "怎么{subject}{condition}还不更新，能{action}不？",
      "遇到{subject}{condition}的情况，要怎么{action}？",
      "我的快递是{subject}，现在{condition}，能否加急{action}？",
    ],
    answerTemplate:
      "若{subject}出现{condition}，我们将立即向承运商发起异常件工单并{action}，通常 24 小时内会更新最新中转轨迹并反馈核查结果。",
  },
  {
    category: "logistics",
    intentTag: "delivery_address_change",
    subjects: [
      "在途快递包裹", "待揽收包裹", "已到本地网点的件", "代收点寄存包裹",
      "发往公司地址的件", "学校驿站包裹", "跨省运输包裹", "同城配送件",
    ],
    conditions: [
      "已经出库在路上", "不小心填错门牌号", "收件人临时出差", "公司周末无人收货",
      "驿站不支持大件", "需要转寄新城市", "预留手机号停机", "收货人姓名写错",
    ],
    actions: ["改派新地址", "修改收件电话", "转寄包裹", "联系派件员改约"],
    questionTemplates: [
      "{subject}因为{condition}，现在还能{action}吗？",
      "请问{subject}已经{condition}，怎么申请{action}？",
      "我的{subject}遇到{condition}，可以帮忙{action}不？",
      "快递是{subject}但{condition}，能线上{action}吗？",
      "如果{subject}{condition}，能否联系物流{action}？",
      "急问，{subject}目前{condition}，来得及{action}吗？",
    ],
    answerTemplate:
      "对于{subject}遇到{condition}的情况，若尚未完成最终派送，我们可协助登记转寄工单为您{action}；跨省改地址可能会产生少量转寄运费（以快递核实为准）。",
  },
  {
    category: "logistics",
    intentTag: "abnormal_sign",
    subjects: [
      "生鲜水果包裹", "数码电子包裹", "贵重美妆包裹", "普通日用包裹",
      "多箱拆分订单", "易碎陶瓷包裹", "文件合同包裹", "服饰箱包包裹",
    ],
    conditions: [
      "显示已签收但本人没收到", "被误投到其他楼栋", "快递柜未发取件码",
      "只收到其中1个包裹", "未通知直接放门口", "签收人名字不认识",
      "驿站查不到入库记录", "显示他人代签",
    ],
    actions: ["核实签收底单", "找回包裹", "确认分箱单号", "联系派件员核对"],
    questionTemplates: [
      "我的{subject}{condition}，请问怎么{action}？",
      "{subject}竟然{condition}了，能帮我{action}吗？",
      "遇到{subject}{condition}，现在该怎么{action}？",
      "系统提示{subject}{condition}，麻烦协助{action}。",
      "为什么{subject}会{condition}？能尽快{action}不？",
      "急！{subject}{condition}，请帮我{action}。",
    ],
    answerTemplate:
      "遇到{subject}{condition}时请勿担心，请先核对快递柜或门卫/驿站；同时我们会为您调取派件员签收定位及底单，24 小时内协助{action}。",
  },
  {
    category: "logistics",
    intentTag: "courier_preference",
    subjects: [
      "易碎玻璃制品", "急用商务礼盒", "偏远乡镇包裹", "大重量米面粮油",
      "高价值数码产品", "夜间配送订单", "校园快递点包裹", "冷链生鲜订单",
    ],
    conditions: [
      "家里只有顺丰能进小区", "某家快递经常丢件", "需要周末送货上门",
      "不能放菜鸟驿站", "需要提前电话预约", "超重需要送上楼",
      "想要隐私面单发货", "需要保价运输",
    ],
    actions: ["指定顺丰发货", "备注送货上门", "更换承运快递", "开启隐私保护"],
    questionTemplates: [
      "买{subject}时因为{condition}，可以{action}吗？",
      "请问购买{subject}如果{condition}，支持{action}不？",
      "我的{subject}{condition}，下单能不能{action}？",
      "关于{subject}配送，{condition}的话怎么{action}？",
      "想问下{subject}在{condition}时能免费{action}吗？",
      "针对{subject}，遇到{condition}怎么申请{action}？",
    ],
    answerTemplate:
      "针对{subject}的{condition}需求，您可在下单页备注配送要求或联系客服登记；符合条件的大件或指定区域包裹将为您优先{action}。",
  },
  {
    category: "logistics",
    intentTag: "international_customs",
    subjects: [
      "保税仓进口商品", "海外直邮订单", "香港仓直发商品", "跨境美妆包裹",
      "进口奶粉订单", "海外保健品包裹", "跨境奢品包裹", "进口母婴件",
    ],
    conditions: [
      "清关需要身份证", "收件人与付款人不一致", "超过个人年度额度",
      "被海关抽检", "提示清关异常", "关税预缴核对",
      "运单号国际段未更新", "周末海关休息",
    ],
    actions: ["补充清关资料", "查询清关进展", "核对税费明细", "完成实名认证"],
    questionTemplates: [
      "{subject}提示{condition}，请问怎么{action}？",
      "买了{subject}但{condition}，要如何{action}？",
      "请问{subject}在{condition}后多久能放行，需要{action}吗？",
      "我的{subject}遇到{condition}，去哪里{action}？",
      "跨境订单{subject}目前{condition}，能指导我{action}吗？",
      "{subject}因为{condition}卡住了，怎么快速{action}？",
    ],
    answerTemplate:
      "跨境{subject}在{condition}时需严格遵守海关实名规范，请确保下单人、支付人与身份证信息一致，并按短信或订单页提示{action}即可恢复清关。",
  },
  {
    category: "logistics",
    intentTag: "package_intercept",
    subjects: [
      "刚发出的快递", "尚未派送的包裹", "在途换货包裹", "误拍的重复订单",
      "地址填错的包裹", "临时不想要的商品", "未拆封拒收件", "已到达当地的包裹",
    ],
    conditions: [
      "还没到收货城市", "刚出仓库揽收", "已经安排派件员",
      "联系不到快递员", "想直接原路退回", "需要拦截改退",
      "部分商品想拦截", "订单已自动确认",
    ],
    actions: ["申请物流拦截", "直接当面拒收", "拦截后退款", "通知网点退回"],
    questionTemplates: [
      "{subject}现在{condition}，能帮我{action}吗？",
      "请问{subject}处于{condition}时怎么{action}？",
      "不小心买错了{subject}且{condition}，可以{action}不？",
      "我的{subject}目前{condition}，还能成功{action}吗？",
      "如果想对{condition}的{subject}{action}，流程是什么？",
      "{subject}在{condition}的情况下，{action}要扣运费吗？",
    ],
    answerTemplate:
      "{subject}在{condition}阶段可由我们向快递网点下发拦截指令尝试{action}；若网点无法拦截，派件时您也可直接{action}，退回仓后将自动触发退款。",
  },

  // ================= refund (7 个子场景) =================
  {
    category: "refund",
    intentTag: "seven_day_return",
    subjects: [
      "未拆封护肤品", "吊牌完好服饰", "试穿未出门鞋靴", "家用小家电",
      "未激活数码产品", "包装完好书籍", "未拆封食品礼盒", "家居日用百货",
    ],
    conditions: [
      "签收3天内不喜欢", "尺码买大了不合适", "颜色和想象有色差",
      "家人重复购买了", "拆了外快递箱但内包装完好", "赠品也保留完整",
      "使用了平台满减券", "已确认收货但未过7天",
    ],
    actions: ["七天无理由退货", "申请换货换码", "办理退货退款", "预约上门取件"],
    questionTemplates: [
      "{subject}在{condition}的情况下，能申请{action}吗？",
      "请问刚买的{subject}因为{condition}，怎么办理{action}？",
      "我的{subject}满足{condition}，可以支持{action}不？",
      "关于{subject}，如果{condition}的话怎么{action}？",
      "买回来的{subject}{condition}，申请{action}有什么要求？",
      "{subject}{condition}，现在可以在订单里点{action}吗？",
    ],
    answerTemplate:
      "{subject}在{condition}且不影响二次销售的前提下支持{action}；您可在订单页选择“退货退款/换货”，系统将自动安排上门取件或提供寄回地址。",
  },
  {
    category: "refund",
    intentTag: "return_shipping_fee",
    subjects: [
      "普通退货订单", "质量问题退货", "尺码换货订单", "发错货退回订单",
      "运费险保障订单", "大件家电退货", "多件退一件订单", "拒收退回订单",
    ],
    conditions: [
      "商品本身有瑕疵", "单纯个人不喜欢", "已购买退货运费险",
      "商家发错颜色规格", "超过运费险首重", "偏远地区退回",
      "包邮商品部分退货", "平台补贴免邮订单",
    ],
    actions: ["报销运费", "理赔运费险", "由商家承担运费", "确认运费承担方"],
    questionTemplates: [
      "{subject}如果是{condition}，退货运费谁出，能{action}吗？",
      "请问{subject}遇到{condition}怎么{action}？",
      "办理{subject}时因为{condition}，运费可以{action}不？",
      "我的{subject}属于{condition}，请问如何{action}？",
      "关于{subject}在{condition}下的运费规则，支持{action}吗？",
      "为什么{subject}{condition}还要垫付运费，之后会{action}吗？",
    ],
    answerTemplate:
      "{subject}在{condition}时：属于商品质量或错发问题将全额{action}；个人原因退货若有运费险，退货签收后 72 小时内系统自动{action}至原账户，超重部分由买家承担。",
  },
  {
    category: "refund",
    intentTag: "refund_arrival_time",
    subjects: [
      "未发货仅退款", "退货签收后退款", "极速退款订单", "价保差价退款",
      "银行卡支付退款", "微信支付宝退款", "花呗分期退款", "礼品卡抵扣退款",
    ],
    conditions: [
      "提交申请刚通过", "退货快递显示已签收", "系统提示退款中",
      "跨周末或节假日", "原支付银行卡已注销", "部分商品退款",
      "金额超过2000元", "使用了组合优惠券",
    ],
    actions: ["原路退回账户", "查询退款进度", "核实到账时间", "确认到账金额"],
    questionTemplates: [
      "{subject}目前{condition}，大概多久能{action}？",
      "请问{subject}在{condition}后几天可以{action}？",
      "我的{subject}已经{condition}，怎么还没看到{action}？",
      "办理了{subject}且{condition}，去哪里{action}？",
      "想问下{subject}如果{condition}，能顺利{action}吗？",
      "{subject}{condition}的话，最晚几个工作日{action}？",
    ],
    answerTemplate:
      "{subject}在{condition}并审核通过后，零钱/余额一般 1-2 个工作日{action}，银行卡或信用卡受银行清算周期影响通常需 3-7 个工作日。",
  },
  {
    category: "refund",
    intentTag: "exchange_process",
    subjects: [
      "尺码偏小外套", "尺码偏大运动鞋", "功能故障小家电", "发错型号配件",
      "颜色发错礼盒", "配件缺失商品", "轻微压痕图书", "不合适宠物用品",
    ],
    conditions: [
      "想换同款其他尺码", "需要换同价位其他颜色", "商品吊牌未剪",
      "外包装稍微压扁", "想要先发新货后退旧货", "目标尺码只剩最后几件",
      "签收还没超过15天", "已经使用过一次发现异常",
    ],
    actions: ["申请同款换货", "预约取旧发新", "锁定换货库存", "办理补发配件"],
    questionTemplates: [
      "收到的{subject}因为{condition}，可以{action}吗？",
      "请问{subject}{condition}的话，怎么申请{action}？",
      "我的{subject}满足{condition}，支持快速{action}不？",
      "想对{subject}进行{action}，但目前{condition}怎么办？",
      "关于{subject}在{condition}下的换货，能直接{action}吗？",
      "买了{subject}发现{condition}，{action}大概需要几天？",
    ],
    answerTemplate:
      "对于{subject}{condition}的情况，您可在订单详情点击“申请换货”，写明目标规格后仓库会在收到退回包裹或核实快递单号后为您优先{action}。",
  },
  {
    category: "refund",
    intentTag: "price_protection",
    subjects: [
      "数码家电订单", "护肤彩妆订单", "品牌服饰订单", "家居用品订单",
      "日用百货订单", "电脑外设订单", "运动装备订单", "母婴奶粉订单",
    ],
    conditions: [
      "签收7天内官方降价", "刚买完遇到大促满减", "平台发放了大额补贴券",
      "同款商品页面直接降价", "保价期内出现差价", "百亿补贴临时降价",
      "凑单后发现单价变低", "预售尾款比现货还贵",
    ],
    actions: ["申请一键价保", "退还买贵差价", "核对价保规则", "领取差价补偿"],
    questionTemplates: [
      "{subject}在{condition}，可以{action}吗？",
      "刚买的{subject}遇到{condition}，怎么{action}？",
      "请问{subject}{condition}支持{action}不？",
      "我的{subject}因为{condition}买贵了，能申请{action}吗？",
      "关于{subject}的价保政策，{condition}符合{action}条件吗？",
      "遇到{subject}{condition}，在哪里提交{action}？",
    ],
    answerTemplate:
      "若{subject}标注支持“价保服务”且在{condition}范围内，您可在“我的订单-价格保护”中发起校验，系统核实差价无误后将自动{action}。",
  },
  {
    category: "refund",
    intentTag: "special_non_returnable",
    subjects: [
      "定制刻字礼物", "已拆封贴身内衣", "生鲜易腐水果", "已激活软件激活码",
      "已使用虚拟会员卡", "裁剪定制窗帘", "已拆塑封书籍", "临期清仓特卖品",
    ],
    conditions: [
      "没有质量问题想退货", "拆封后发现不喜欢", "收货超过48小时",
      "包装盒已丢弃", "已经绑定个人账号", "发现轻微质量瑕疵",
      "商家发错了定制内容", "运输途中变质坏果",
    ],
    actions: ["特殊售后处理", "按比例退款", "协商退换货", "申请质量赔付"],
    questionTemplates: [
      "{subject}如果{condition}，还能办理{action}吗？",
      "请问购买了{subject}且{condition}，支持{action}不？",
      "我的{subject}属于{condition}，能申请{action}吗？",
      "关于{subject}，遇到{condition}怎么争取{action}？",
      "想确认下{subject}在{condition}下能否{action}？",
      "{subject}提示不支持无理由，但现在{condition}可以{action}吗？",
    ],
    answerTemplate:
      "{subject}因商品属性通常不支持无理由退换；但若属于{condition}中的质量、变质或错发问题，请保留凭证照片提交售后，我们将为您安排{action}。",
  },
  {
    category: "refund",
    intentTag: "partial_and_coupon_refund",
    subjects: [
      "满300减50凑单订单", "买二送一活动订单", "多件组合套装订单", "使用平台红包订单",
      "跨店满减合并订单", "加价购换购订单", "积分加现金订单", "多规格拼单订单",
    ],
    conditions: [
      "只退其中一件商品", "退货后不满足满减门槛", "部分商品缺货退款",
      "赠品未拆封留存", "优惠券已过期才退款", "主商品退货但赠品已用",
      "系统按比例分摊优惠", "想保留赠品退主商品",
    ],
    actions: ["计算实退金额", "退回可用优惠券", "单独退单件商品", "核对优惠分摊"],
    questionTemplates: [
      "{subject}如果{condition}，怎么{action}？",
      "请问{subject}{condition}时，能{action}吗？",
      "我的{subject}现在{condition}，退款会如何{action}？",
      "办理{subject}退货遇到{condition}，支持{action}不？",
      "为什么{subject}在{condition}后退的钱变少了，怎么{action}？",
      "关于{subject}{condition}的问题，平台怎样{action}？",
    ],
    answerTemplate:
      "{subject}在{condition}时，系统会按各商品实付金额比例分摊优惠并{action}；整单全额退且券仍在有效期内时，优惠券将同步退回账户。",
  },

  // ================= complaint (7 个子场景) =================
  {
    category: "complaint",
    intentTag: "damaged_in_transit",
    subjects: [
      "玻璃水杯礼盒", "陶瓷餐具套装", "精密数码屏幕", "瓶装精华液",
      "精装礼品礼盒", "易碎零食罐", "实木家具边角", "显示器外壳",
    ],
    conditions: [
      "拆开外箱发现内部碎裂", "快递暴力运输导致变形", "液体漏液污染全部商品",
      "边角磕碰掉漆严重", "内胆破裂无法使用", "外包装湿透发霉",
      "配件折断", "明显摔裂痕迹",
    ],
    actions: ["申请破损赔付", "要求补发新货", "投诉物流暴力运输", "办理全额退款"],
    questionTemplates: [
      "收到的{subject}居然{condition}，我要{action}！",
      "太差劲了，{subject}拆开发现{condition}，怎么{action}？",
      "买来的{subject}{condition}，请问如何{action}？",
      "包裹签收后看到{subject}{condition}，必须{action}！",
      "快递把{subject}弄得{condition}，我要投诉并{action}。",
      "刚收到{subject}就{condition}了，请尽快安排{action}！",
    ],
    answerTemplate:
      "非常抱歉让您收到受损的{subject}（{condition}）。请您先拍摄外包装面单与商品破损照片，提交售后工单后我们将优先为您办理{action}或免退仅补发。",
  },
  {
    category: "complaint",
    intentTag: "missing_or_wrong_item",
    subjects: [
      "多件日用订单", "护肤正装与小样", "电器主机与配件", "服饰套装订单",
      "买一赠一订单", "双拼食品礼盒", "定制配件包", "办公耗材多盒装",
    ],
    conditions: [
      "少发了关键配件", "数量少了一半", "收到完全无关的另一款商品",
      "承诺的赠品完全没发", "尺码和型号全发错了", "礼盒里少了一瓶",
      "说明书和保修卡缺失", "包装袋里面是空的",
    ],
    actions: ["要求补发漏发件", "投诉仓库漏发错发", "申请少件赔付", "安排退换赔礼"],
    questionTemplates: [
      "收到的{subject}竟然{condition}，我要{action}！",
      "怎么回事，{subject}拆箱发现{condition}，赶紧{action}！",
      "投诉！{subject}{condition}，请客服立刻帮我{action}。",
      "买了{subject}结果{condition}，要怎么{action}？",
      "{subject}严重货不对板且{condition}，要求{action}！",
      "拆开{subject}时{condition}，有开箱视频怎么{action}？",
    ],
    answerTemplate:
      "给您添麻烦了！针对{subject}{condition}的情况，请提供快递面单与箱内实物照片（有开箱视频更好），核实仓库出库重量后我们将第一时间为您{action}。",
  },
  {
    category: "complaint",
    intentTag: "counterfeit_or_quality",
    subjects: [
      "品牌运动鞋", "高端护肤面霜", "蓝牙降噪耳机", "纯棉床上四件套",
      "进口保健胶囊", "电动剃须刀", "大牌香水礼盒", "儿童益智玩具",
    ],
    conditions: [
      "做工粗糙线头满天飞", "有刺鼻劣质塑料味", "防伪码根本扫不出来",
      "用了一次就彻底坏了", "和专柜正品明显不一样", "屏幕闪烁且严重发热",
      "洗一次就严重褪色缩水", "包装印刷模糊像假货",
    ],
    actions: ["举报商品质量问题", "申请假一赔三核查", "要求质量退赔", "提交质检投诉"],
    questionTemplates: [
      "这批{subject}质量太差，{condition}，我要{action}！",
      "严重怀疑{subject}有问题，{condition}，必须{action}！",
      "投诉商家售卖劣质{subject}，刚用就{condition}，要求{action}。",
      "收到的{subject}{condition}，完全没法用，怎么{action}？",
      "刚拆封{subject}就发现{condition}，请问如何{action}？",
      "你们卖的{subject}竟然{condition}，请给说法并{action}！",
    ],
    answerTemplate:
      "我们高度重视您对{subject}（{condition}）的反馈，平台严禁劣质及侵权商品。请您上传细节图或检测凭证，品控专员将在 24 小时内介入核实并为您办理{action}。",
  },
  {
    category: "complaint",
    intentTag: "service_attitude_complaint",
    subjects: [
      "在线客服接待", "售后专员处理", "快递派件员", "安装师傅上门",
      "仓库客服电话", "门店自提服务", "退换货审核员", "投诉回访专员",
    ],
    conditions: [
      "态度极其敷衍不耐烦", "承诺24小时回复却失联", "多次推诿踢皮球",
      "未经允许直接关闭工单", "说话语气非常生硬", "随意挂断客户电话",
      "拒绝合理售后诉求", "擅自把包裹扔在楼下",
    ],
    actions: ["投诉相关服务人员", "要求主管介入处理", "升级高级投诉工单", "申请服务补偿"],
    questionTemplates: [
      "刚才的{subject}{condition}，我要{action}！",
      "服务体验太差了，{subject}居然{condition}，怎么{action}？",
      "强烈投诉！{subject}{condition}，请立刻{action}！",
      "遇到{subject}{condition}的情况，在哪能{action}？",
      "对这次{subject}非常不满，{condition}，必须{action}！",
      "{subject}一直{condition}，根本不解决问题，我要{action}！",
    ],
    answerTemplate:
      "非常抱歉{subject}{condition}给您带来糟糕体验！我们已记录您的反馈并支持随时为您{action}，投诉监督团队会在 2 小时内复核会话记录并向您致歉整改。",
  },
  {
    category: "complaint",
    intentTag: "delay_compensation",
    subjects: [
      "承诺次日达订单", "48小时必发订单", "节日送礼急单", "预约上门安装服务",
      "生鲜冷链急送单", "高价加急空运单", "预售承诺发货单", "大促保时效订单",
    ],
    conditions: [
      "超时5天仍未发货", "耽误了重要送礼日期", "承诺赔付券一直没到账",
      "虚假发货只打单不揽收", "超期未送达导致商品变质", "反复改期放鸽子",
      "超过约定安装时间三天", "客服虚假承诺发货时间",
    ],
    actions: ["申请延迟发货赔付", "投诉虚假发货", "要求违约补偿", "升级加急催发"],
    questionTemplates: [
      "这个{subject}已经{condition}，我要{action}！",
      "商家太不守信了，{subject}{condition}，怎么{action}？",
      "投诉虚假宣传，{subject}明明承诺好却{condition}，要求{action}！",
      "{subject}严重超时且{condition}，请问如何{action}？",
      "等{subject}等得火大，现在{condition}，必须{action}！",
      "针对{subject}{condition}的问题，平台怎么给予{action}？",
    ],
    answerTemplate:
      "非常理解您等待{subject}（{condition}）的焦急心情。核实超期或虚假发货属实后，我们将按平台时效保障规则为您发放违约赔付红包并{action}。",
  },
  {
    category: "complaint",
    intentTag: "dispute_and_chargeback",
    subjects: [
      "售后退款金额", "运费险拒赔结果", "定金扣除争议", "换货差价扣款",
      "自动续费扣款", "商家拒绝退货理由", "折旧费扣除争议", "保价驳回结果",
    ],
    conditions: [
      "被商家无理驳回申请", "未经过同意直接扣款", "强行扣除高额折旧费",
      "明明符合条件却判我责任", "莫名其妙被自动扣费", "退款少退了运费",
      "举证充分依然判定失败", "申诉入口打不开",
    ],
    actions: ["申请平台官方介入", "发起交易争议仲裁", "投诉乱扣费行为", "要求退还争议款项"],
    questionTemplates: [
      "我对{subject}非常不满，{condition}，我要{action}！",
      "太不公平了，{subject}{condition}，怎么{action}？",
      "投诉！{subject}居然{condition}，请平台{action}！",
      "遇到{subject}{condition}该怎么办，如何{action}？",
      "商家恶意操作导致{subject}{condition}，必须{action}！",
      "{subject}判定不合理且{condition}，请帮我{action}！",
    ],
    answerTemplate:
      "遇到{subject}{condition}的情况，您可在售后单详情点击“申请平台客服介入”并上传聊天与物流凭证，平台争议处理专员将在 24 小时内公正核查并为您{action}。",
  },
  {
    category: "complaint",
    intentTag: "safety_privacy_escalation",
    subjects: [
      "收到的食品包装", "购买的家电插座", "快递面单信息", "账号异常订单",
      "护肤品成分使用", "儿童玩具小零件", "商家骚扰短信", "冒充客服诈骗电话",
    ],
    conditions: [
      "发现过期或异物", "通电后冒烟有安全隐患", "疑似个人隐私被泄露",
      "使用后皮肤严重过敏红肿", "存在锋利毛刺划伤手", "收到威胁好评返现骚扰",
      "怀疑账号被盗刷下单", "发现产品无3C认证",
    ],
    actions: ["举报安全隐患", "投诉隐私泄露", "要求紧急下架核查", "申请安全赔付专员介入"],
    questionTemplates: [
      "严重警告！{subject}{condition}，我要{action}！",
      "太危险了，{subject}竟然{condition}，必须立刻{action}！",
      "投诉举报！{subject}发生{condition}，请马上{action}！",
      "碰到{subject}{condition}这种严重问题，如何{action}？",
      "我的{subject}出现{condition}，要求官方严肃{action}！",
      "极其恶劣，{subject}{condition}，请立刻转专员{action}！",
    ],
    answerTemplate:
      "您反馈的{subject}（{condition}）涉及安全与隐私红线，我们高度重视并已升级至高级安全专员通道；请先停止使用相关商品并保留证据，我们将立即{action}。",
  },
];

const COLLOQUIAL_PREFIXES = [
  "",
  "你好，",
  "您好，",
  "麻烦问一下，",
  "不好意思打扰下，",
  "请问下，",
  "在吗？",
  "亲，",
  "客服在吗，",
  "急问一下，",
  "想咨询个事，",
  "帮我看下，",
  "问一下哈，",
  "说实话，",
  "麻烦帮忙查下，",
] as const;

const COLLOQUIAL_SUFFIXES = [
  "",
  "谢谢",
  "谢谢哈",
  "麻烦尽快看下",
  "在线等挺急的",
  "麻烦给个准信",
  "可以吗",
  "行不行啊",
  "怎么处理比较快",
  "拜托了",
  "帮我处理一下呗",
  "感谢",
] as const;

const SYNONYM_REPLACEMENTS: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["请问", ["想问下", "咨询下", "问下"]],
  ["可以", ["能", "能不能", "支不支持"]],
  ["怎么", ["咋", "要如何", "该怎么"]],
  ["什么时候", ["啥时候", "大概几点", "多久"]],
  ["大概多久", ["要几天", "一般几天", "需要多长时间"]],
  ["非常", ["特别", "实在太", "极其"]],
  ["要求", ["必须", "麻烦给我", "希望"]],
  ["投诉", ["我要举报", "我要反馈", "差评投诉"]],
  ["帮忙", ["协助", "帮我", "尽快帮忙"]],
  ["商品", ["宝贝", "东西", "货品"]],
  ["包裹", ["快递", "件", "货"]],
  ["订单", ["单子", "这笔单"]],
  ["退款", ["退钱", "把钱退回来"]],
  ["退货", ["退回去", "把货退了"]],
  ["破损", ["碎了", "摔坏了", "压坏了"]],
  ["优惠券", ["满减券", "代金券", "券"]],
  ["发票", ["报销发票", "电子票"]],
  ["客服", ["人工", "专员"]],
];

function fillTemplate(
  template: string,
  slots: { subject: string; condition: string; action: string }
): string {
  return template
    .replaceAll("{subject}", slots.subject)
    .replaceAll("{condition}", slots.condition)
    .replaceAll("{action}", slots.action);
}

function mutateColloquially(
  question: string,
  rng: () => number,
  variantIndex: number
): { text: string; rule: string } {
  let text = question;
  const rules: string[] = [];

  // 1. 替换 1~2 个同义词
  const candidates = SYNONYM_REPLACEMENTS.filter(([src]) => text.includes(src));
  if (candidates.length > 0) {
    const pickCount = candidates.length > 1 && rng() > 0.45 ? 2 : 1;
    const shuffled = [...candidates].sort(() => rng() - 0.5);
    for (let i = 0; i < pickCount; i += 1) {
      const [src, targets] = shuffled[i];
      const replacement = targets[Math.floor(rng() * targets.length)];
      text = text.replace(src, replacement);
    }
    rules.push("synonym");
  }

  // 2. 去掉或调整结尾标点
  text = text.replace(/[？?！!。.]$/, "");

  // 3. 添加口语前缀
  const prefixIdx =
    variantIndex === 0
      ? 1 + Math.floor(rng() * (COLLOQUIAL_PREFIXES.length - 1))
      : Math.floor(rng() * COLLOQUIAL_PREFIXES.length);
  const prefix = COLLOQUIAL_PREFIXES[prefixIdx];
  if (prefix) {
    text = `${prefix}${text}`;
    rules.push("prefix");
  }

  // 4. 添加口语后缀或句末助词
  const suffixIdx = Math.floor(rng() * COLLOQUIAL_SUFFIXES.length);
  const suffix = COLLOQUIAL_SUFFIXES[suffixIdx];
  if (suffix) {
    const punct = rng() > 0.5 ? "，" : " ";
    text = `${text}${punct}${suffix}`;
    rules.push("suffix");
  }

  // 5. 句末标点变体
  const endPuncts = ["？", "?", "！", "", "～", "。"];
  const endPunct = endPuncts[Math.floor(rng() * endPuncts.length)];
  text = `${text}${endPunct}`;

  if (rules.length === 0) {
    rules.push("punct");
  }

  return { text: text.trim(), rule: rules.join("+") };
}

/**
 * 确定性多维槽位组合 + 口语同义改写数据合成引擎
 * - 先生成 baseCount 条严格去重的基础母本语料（四大类均衡）
 * - 再基于母本进行口语/同义/前缀/后缀组合合成扩增至 targetCount 条严格去重样本
 */
export function synthesizeQaDataset(
  options?: SynthDatasetOptions
): SynthDatasetResult {
  const start = Date.now();
  const baseCount = options?.baseCount ?? 10_000;
  const targetCount = Math.max(baseCount, options?.targetCount ?? 100_000);
  const seed = options?.seed ?? 42;
  const rng = createSeededRng(seed);

  const categories: StandardCategory[] = [
    "consultation",
    "logistics",
    "refund",
    "complaint",
  ];
  const basePerCategory = Math.floor(baseCount / categories.length);

  // 为每个类别构建全量槽位组合池
  const poolsByCategory = new Map<
    StandardCategory,
    Array<{
      intentTag: string;
      question: string;
      answer: string;
    }>
  >();

  for (const cat of categories) {
    poolsByCategory.set(cat, []);
  }

  for (const bp of SCENARIO_BLUEPRINTS) {
    const pool = poolsByCategory.get(bp.category)!;
    for (const subject of bp.subjects) {
      for (const condition of bp.conditions) {
        for (const action of bp.actions) {
          for (const qTpl of bp.questionTemplates) {
            const slots = { subject, condition, action };
            pool.push({
              intentTag: bp.intentTag,
              question: fillTemplate(qTpl, slots),
              answer: fillTemplate(bp.answerTemplate, slots),
            });
          }
        }
      }
    }
  }

  // 洗牌每个类别池并去重抽取 baseCount 条基础语料
  const seenQuestions = new Set<string>();
  const baseRecords: SyntheticQaRecord[] = [];
  let baseSeq = 1;

  for (let cIdx = 0; cIdx < categories.length; cIdx += 1) {
    const cat = categories[cIdx];
    const pool = poolsByCategory.get(cat)!;
    for (let i = pool.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }

    const quota =
      cIdx === categories.length - 1
        ? baseCount - baseRecords.length
        : basePerCategory;
    let taken = 0;
    for (const item of pool) {
      if (taken >= quota) break;
      const key = item.question.toLowerCase();
      if (seenQuestions.has(key)) continue;
      seenQuestions.add(key);

      const id = `base-${String(baseSeq).padStart(6, "0")}`;
      baseSeq += 1;
      baseRecords.push({
        id,
        baseKnowledgeId: id,
        sourceType: "base",
        rawCategory: cat,
        intentTag: item.intentTag,
        question: item.question,
        answer: item.answer,
      });
      taken += 1;
    }
  }

  if (baseRecords.length < baseCount) {
    throw new Error(
      `基础语料组合池不足：期望 ${baseCount} 条，实际仅生成 ${baseRecords.length} 条`
    );
  }

  // 合成扩增阶段：在保持类别均衡的前提下为每条基础语料生成多个口语/同义变体
  const allRecords: SyntheticQaRecord[] = [...baseRecords];
  const categoryCounts: Record<StandardCategory, number> = {
    consultation: 0,
    logistics: 0,
    refund: 0,
    complaint: 0,
  };
  for (const r of baseRecords) {
    categoryCounts[r.rawCategory] += 1;
  }

  let synSeq = 1;
  let round = 0;
  const maxRounds = 40;

  while (allRecords.length < targetCount && round < maxRounds) {
    for (const base of baseRecords) {
      if (allRecords.length >= targetCount) break;
      let generated = false;
      for (let attempt = 0; attempt < 8; attempt += 1) {
        const mutated = mutateColloquially(base.question, rng, round + attempt);
        const key = mutated.text.toLowerCase();
        if (seenQuestions.has(key)) continue;
        seenQuestions.add(key);

        allRecords.push({
          id: `syn-${String(synSeq).padStart(6, "0")}`,
          baseKnowledgeId: base.baseKnowledgeId,
          sourceType: "synthetic",
          rawCategory: base.rawCategory,
          intentTag: base.intentTag,
          question: mutated.text,
          answer: base.answer,
          variantRule: mutated.rule,
        });
        categoryCounts[base.rawCategory] += 1;
        synSeq += 1;
        generated = true;
        break;
      }
      if (!generated) {
        continue;
      }
    }
    round += 1;
  }

  if (allRecords.length < targetCount) {
    throw new Error(
      `合成扩增未达目标：期望 ${targetCount} 条，实际生成 ${allRecords.length} 条`
    );
  }

  return {
    baseRecords,
    allRecords,
    stats: {
      baseCount: baseRecords.length,
      totalCount: allRecords.length,
      syntheticCount: allRecords.length - baseRecords.length,
      categoryCounts,
      durationMs: Date.now() - start,
    },
  };
}
