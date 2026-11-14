import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { QaCategory, QaSample, StandardCategory } from "@ts-qa/shared";
import {
  createSeededRng,
  mapToStandardCategory,
  synthesizeQaDataset,
} from "../packages/cleaner/src/index.ts";
import {
  CharNgramNaiveBayesClassifier,
  evaluateIntentClassifier,
} from "../packages/trainer/src/index.ts";

interface HardTemplate {
  category: StandardCategory;
  patterns: readonly string[];
  slotsA: readonly string[];
  slotsB: readonly string[];
  answer: string;
}

/**
 * 专门强化复合 N-gram 朴素贝叶斯（反关键词干扰 + 隐式语义 + 长文本共现）的训练集模板
 */
const BAYES_ENHANCED_TEMPLATES: readonly HardTemplate[] = [
  // consultation：含售后/物流词干扰，但核心是售前规则/账户/开票/活动咨询
  {
    category: "consultation",
    patterns: [
      "我不打算{distractor}，就是下单前想确认下{topic}{detail}",
      "还没付款也没有要{distractor}，麻烦问下{topic}{detail}",
      "先不{distractor}哈，我想提前了解下{topic}{detail}",
      "请问买之前如果考虑{topic}，{detail}需要怎么操作？",
      "帮我查下账户规则：{topic}在{detail}时能正常用吗？",
    ],
    slotsA: [
      "会员积分抵扣", "新人满减券", "电子普通发票", "企业专票抬头",
      "绑定手机号换绑", "分期免息资格", "跨店凑单门槛", "生日礼券领取",
    ],
    slotsB: [
      "过期前能不能补领", "结算页面怎么勾选", "税号填错了怎么改",
      "原号码停用了咋验证", "哪些特价款不参与", "周末客服几点下班",
    ],
    answer: "为您查询到对应账户与售前规则，请按页面活动说明与账户中心指引操作即可。",
  },
  // logistics：口语化无“物流/快递”字眼，或含“不退货/不投诉”干扰
  {
    category: "logistics",
    patterns: [
      "暂时不{distractor}，就想问下前天拍的{item}{issue}",
      "先别办{distractor}，我那个{item}显示{issue}，能催一下吗",
      "帮忙看下单子进度，{item}一直{issue}，啥时候能送到门口",
      "这单{item}已经两天了还是{issue}，帮我联系网点看下到哪了",
    ],
    slotsA: [
      "生鲜水果礼盒", "小家电包裹", "换季衣服", "猫粮狗粮",
      "电脑外设", "办公用品箱", "母婴纸尿裤", "陶瓷水杯",
    ],
    slotsB: [
      "停在分拨中心不动了", "显示别人代签但我家里没人拿",
      "驿站查不到取件码", "只发出来半截路就没更新了",
      "派件员放错了楼栋门卫", "填错了门牌号需要转寄",
    ],
    answer: "已为您核实承运网点与中转状态，我们将发起异常件催派并协助核对签收/转寄进度。",
  },
  // refund：隐式售后诉求（不直接说“退货/退款”或含“不投诉”干扰）
  {
    category: "refund",
    patterns: [
      "我不打算{distractor}，刚买的{item}因为{reason}，能把钱退回原卡吗",
      "先不{distractor}，这件{item}{reason}，我想换个大一号的码怎么弄",
      "前两天买的{item}刚降了50块，符合{reason}把差价补给我吗",
      "收到{item}试了一下{reason}，吊牌都在想寄回去重拍怎么走流程",
    ],
    slotsA: [
      "连衣裙", "运动跑鞋", "护肤精华", "无线耳机",
      "机械键盘", "电饭煲", "保温杯", "行李箱",
    ],
    slotsB: [
      "尺码明显偏小穿不上", "家里人买重了一单", "官方七天内降价买贵了",
      "颜色和图上色差太大", "拆开外箱没动内封不想要了", "运费险不知道够不够首重",
    ],
    answer: "您可在订单详情页提交退换货或一键价保申请，符合条件将原路退回款项或安排取旧换新。",
  },
  // complaint：长文本抱怨/隐式质量破损，不直接带“投诉”二字
  {
    category: "complaint",
    patterns: [
      "真是无语透顶，花钱买个{item}结果{bad}，你们平台必须给个说法并赔付！",
      "太气人了，刚拆箱发现{item}{bad}，联系客服还一直敷衍踢皮球，必须严肃处理",
      "这质量也太离谱了吧，{item}用了一天就{bad}，严重怀疑发了劣质翻新货，要求退一赔三",
      "本来急着送人的{item}不仅拖延好几天，收到一看竟然{bad}，体验差到极点！",
    ],
    slotsA: [
      "玻璃红酒杯", "陶瓷餐具", "品牌吹风机", "高档面霜",
      "儿童积木玩具", "蓝牙音箱", "真皮手提包", "不锈钢锅具",
    ],
    slotsB: [
      "里面全碎成渣了而且漏液", "少了一大半关键配件根本没法装",
      "通电直接冒烟有焦糊味太危险", "做工粗糙掉漆严重像山寨货",
      "包装盒湿透发霉还散发异味", "发成完全不相干的廉价商品",
    ],
    answer: "非常抱歉给您带来糟糕体验！请保留实物照片与订单号，我们已为您开启质量/破损优先赔付专员通道。",
  },
];

const DISTRACTORS: Record<StandardCategory, readonly string[]> = {
  consultation: ["退货", "退款", "投诉"],
  logistics: ["退货", "退款", "投诉"],
  refund: ["投诉", "举报"],
  complaint: ["退货"],
};

function generateHardSamples(
  countPerCat: number,
  seed: number,
  datasetVersionId: string,
  split: "train" | "test",
  templateSubset: "train" | "eval"
): QaSample[] {
  const rng = createSeededRng(seed);
  const out: QaSample[] = [];
  const seen = new Set<string>();

  for (const tpl of BAYES_ENHANCED_TEMPLATES) {
    const distractors = DISTRACTORS[tpl.category];
    // 严格隔离训练集与盲测集的 slotsA/slotsB 切片，保证测试集组合从未在训练集出现
    const halfA = Math.floor(tpl.slotsA.length / 2);
    const halfB = Math.floor(tpl.slotsB.length / 2);
    const slotsA =
      templateSubset === "train"
        ? tpl.slotsA.slice(0, halfA + 1)
        : tpl.slotsA.slice(halfA + 1);
    const slotsB =
      templateSubset === "train"
        ? tpl.slotsB.slice(0, halfB + 1)
        : tpl.slotsB.slice(halfB + 1);

    const pool: QaSample[] = [];
    for (const pat of tpl.patterns) {
      for (const a of slotsA) {
        for (const b of slotsB) {
          for (const d of distractors) {
            const q = pat
              .replace("{distractor}", d)
              .replace("{topic}", a)
              .replace("{detail}", b)
              .replace("{item}", a)
              .replace("{issue}", b)
              .replace("{reason}", b)
              .replace("{bad}", b);
            const key = q.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            pool.push({
              id: `${datasetVersionId}-${tpl.category}-${pool.length}`,
              datasetVersionId,
              question: q,
              answer: tpl.answer,
              category: tpl.category,
              split,
            });
          }
        }
      }
    }

    // 洗牌并抽取
    for (let i = pool.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }
    out.push(...pool.slice(0, countPerCat));
  }
  return out;
}

async function main() {
  const repoRoot = resolve(import.meta.dirname, "..");
  const dataDir = resolve(repoRoot, "data");
  mkdirSync(dataDir, { recursive: true });

  // 1. 生成基础合成数据集 (6,000 条) + 复合特征强化样本 (2,000 条) = 8,000 条复合贝叶斯训练集
  const baseSynth = synthesizeQaDataset({
    baseCount: 2000,
    targetCount: 6000,
    seed: 888,
  });
  const synthSamples: QaSample[] = baseSynth.allRecords.map((r) => ({
    id: r.id,
    datasetVersionId: "bayes-train-8k",
    question: r.question,
    answer: r.answer,
    category: r.rawCategory,
    split: "train",
  }));

  const hardTrainSamples = generateHardSamples(
    500,
    999,
    "bayes-hard-train",
    "train",
    "train"
  );
  // 过采样高难度复合样本 3x 增强模型对反干扰词与隐式特征的注意力
  const fullTrain = [
    ...synthSamples,
    ...hardTrainSamples,
    ...hardTrainSamples,
    ...hardTrainSamples,
  ];

  // 2. 生成完全隔离槽位的外部难点盲测集 (200 条，四类各 50 条)
  const hardEvalSamples = generateHardSamples(
    50,
    777,
    "bayes-hard-eval",
    "test",
    "eval"
  );

  const trainPath = resolve(dataDir, "bayes-enhanced-train-8k.jsonl");
  const evalPath = resolve(dataDir, "bayes-hard-eval-200.jsonl");

  writeFileSync(
    trainPath,
    [...synthSamples, ...hardTrainSamples]
      .map((s) =>
        JSON.stringify({
          question: s.question,
          answer: s.answer,
          category: s.category,
        })
      )
      .join("\n") + "\n",
    "utf-8"
  );

  writeFileSync(
    evalPath,
    hardEvalSamples
      .map((s) =>
        JSON.stringify({
          question: s.question,
          answer: s.answer,
          category: s.category,
        })
      )
      .join("\n") + "\n",
    "utf-8"
  );

  // 3. 对比评测：静态关键词规则 vs 复合 N-gram 朴素贝叶斯 (1..3-gram)
  const ruleMetrics = await evaluateIntentClassifier(hardEvalSamples, (q) =>
    mapToStandardCategory(undefined, q)
  );

  const nb2 = new CharNgramNaiveBayesClassifier({
    ngramMin: 1,
    ngramMax: 2,
    alpha: 0.5,
  });
  nb2.fit(fullTrain);
  const nb2Metrics = await evaluateIntentClassifier(hardEvalSamples, (q) =>
    nb2.predict(q)
  );

  const nb3 = new CharNgramNaiveBayesClassifier({
    ngramMin: 1,
    ngramMax: 3,
    alpha: 0.5,
  });
  nb3.fit(fullTrain);
  const nb3Metrics = await evaluateIntentClassifier(hardEvalSamples, (q) =>
    nb3.predict(q)
  );

  console.log(`已生成复合贝叶斯训练集: ${trainPath} (${synthSamples.length + hardTrainSamples.length} 条去重样本)`);
  console.log(`已生成独立难点盲测集: ${evalPath} (${hardEvalSamples.length} 条，槽位完全隔离)\n`);
  console.table({
    "1. 静态关键词规则 (keyword_baseline)": {
      Accuracy: ruleMetrics.accuracy,
      "Macro-F1": ruleMetrics.macroF1,
    },
    "2. 复合贝叶斯 (1..2-gram, alpha=0.5)": {
      Accuracy: nb2Metrics.accuracy,
      "Macro-F1": nb2Metrics.macroF1,
    },
    "3. 复合贝叶斯 (1..3-gram, alpha=0.5)": {
      Accuracy: nb3Metrics.accuracy,
      "Macro-F1": nb3Metrics.macroF1,
    },
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
