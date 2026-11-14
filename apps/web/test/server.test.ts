import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAppServer } from "../src/server.ts";

test("Web HTTP 服务端到端测试：初始化基线、问答、实验复现与人工接管", async () => {
  const tempDir = mkdtempSync(join(tmpdir(), "ts-qa-web-"));
  const server = await createAppServer({
    dataDir: tempDir,
    useOllama: false,
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });

  const addr = server.address() as AddressInfo;
  const baseUrl = `http://127.0.0.1:${addr.port}`;

  try {
    // 1. 首页 HTML 可访问
    const htmlRes = await fetch(`${baseUrl}/`);
    assert.equal(htmlRes.status, 200);
    const html = await htmlRes.text();
    assert.ok(html.includes("培训-客服系统demo"));

    // 2. 启动自动初始化的 v1.0.0 基线应已存在
    const overview1 = (await (await fetch(`${baseUrl}/api/overview`)).json()) as {
      datasetVersions: Array<{ version: string }>;
      trainingJobs: Array<{ jobType: string }>;
    };
    assert.equal(overview1.datasetVersions.length, 1);
    assert.equal(overview1.datasetVersions[0].version, "v1.0.0");
    assert.equal(overview1.trainingJobs.length, 2);

    // 3. 触发新实验（oversample + seed=7 + naive_bayes + ngramMax=2 + alpha=0.5）
    const expRes = (await (
      await fetch(`${baseUrl}/api/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version: "v1.1.0",
          seed: 7,
          balanceStrategy: "oversample",
          splitRatio: { train: 0.7, val: 0.15, test: 0.15 },
          classifierType: "naive_bayes",
          ngramMax: 2,
          alpha: 0.5,
          evalSplit: "test",
        }),
      })
    ).json()) as {
      datasetVersion: { version: string; totalCount: number };
      evalJob: { metrics?: { accuracy: number; macroF1: number; confusionMatrix?: unknown } };
    };
    assert.equal(expRes.datasetVersion.version, "v1.1.0");
    assert.equal(expRes.datasetVersion.totalCount, 100);
    assert.ok(expRes.evalJob.metrics?.confusionMatrix);
    assert.ok(expRes.evalJob.metrics?.ragHitAt3 !== undefined);
    assert.ok((expRes.evalJob.metrics?.macroF1 ?? 0) >= 0.5);

    // 4. 客户问答与投诉转人工
    const chatRes = (await (
      await fetch(`${baseUrl}/api/chat/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: "web-sess-1",
          content: "我要投诉你们的服务，立即转人工！",
        }),
      })
    ).json()) as { ticket?: { id: string } };
    assert.ok(chatRes.ticket?.id);

    // 5. 人工客服认领并回复，并一键收录为标准 QA 样本反哺训练集
    const replyRes = await fetch(`${baseUrl}/api/handoff/reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticketId: chatRes.ticket.id,
        replyContent: "您好，人工客服已接手为您处理。",
      }),
    });
    assert.equal(replyRes.status, 200);

    const promoteRes = await fetch(`${baseUrl}/api/handoff/promote-qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticketId: chatRes.ticket.id }),
    });
    assert.equal(promoteRes.status, 200);

    // 6. 一键自动化调参（跨算法 + 多种子 × 5-Fold CV -> Test 盲测）
    const tuneRes = (await (
      await fetch(`${baseUrl}/api/pipeline/autotune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: 42,
          multiSeed: true,
          splitRatio: { train: 0.7, val: 0.15, test: 0.15 },
        }),
      })
    ).json()) as {
      trials: number;
      bestParams: {
        classifierType: string;
        balanceStrategy: string;
        ngramMax: number;
        alpha: number;
        seedsEvaluated: number;
      };
      finalResult: { evalJob: { metrics?: { macroF1: number } } };
    };
    assert.equal(tuneRes.trials, 48);
    assert.equal(tuneRes.bestParams.seedsEvaluated, 3);
    assert.ok(tuneRes.bestParams.ngramMax >= 1);

    // 7. 审核通过反哺 QA 样本后，支持导出 SFT 微调 JSONL 与移除默认演示语料
    const reviewList = (await (
      await fetch(`${baseUrl}/api/qa-review/list`)
    ).json()) as { items: Array<{ id: string; status: string }> };
    assert.ok(reviewList.items.length >= 1);
    assert.equal(reviewList.items[0].status, "pending_review");

    const decideRes = await fetch(`${baseUrl}/api/qa-review/decide`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: reviewList.items[0].id, decision: "approve" }),
    });
    assert.equal(decideRes.status, 200);

    const exportRes = await fetch(`${baseUrl}/api/dataset/export-finetune`);
    assert.equal(exportRes.status, 200);
    const exportedText = await exportRes.text();
    const firstExported = JSON.parse(exportedText.split("\n")[0]);
    assert.ok(firstExported.instruction && firstExported.input && firstExported.output);

    const removeDefaultRes = await fetch(`${baseUrl}/api/corpus/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ onlyCustom: true }),
    });
    assert.equal(removeDefaultRes.status, 200);

    const ovAfter = (await (await fetch(`${baseUrl}/api/overview`)).json()) as {
      availablePresets: Array<{ label: string; filePath: string }>;
    };
    assert.ok(Array.isArray(ovAfter.availablePresets) && ovAfter.availablePresets.length >= 1);
    const usePresetRes = await fetch(`${baseUrl}/api/corpus/use-preset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filePath: ovAfter.availablePresets[0].filePath,
        replaceCurrent: false,
      }),
    });
    assert.equal(usePresetRes.status, 200);
    assert.ok((tuneRes.finalResult.evalJob.metrics?.macroF1 ?? 0) > 0);

    // 8. 生产化运维与治理闭环：健康探针、监控指标、订单工具与版本回滚
    const healthRes = await fetch(`${baseUrl}/healthz`);
    assert.equal(healthRes.status, 200);
    const readyRes = await fetch(`${baseUrl}/readyz`);
    assert.equal(readyRes.status, 200);

    const orderChatRes = (await (
      await fetch(`${baseUrl}/api/chat/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: "order-test-sess",
          content: "帮我查一下 ORD-1001 物流到哪了",
        }),
      })
    ).json()) as {
      orderInfo?: { orderId: string; carrier: string };
      subIntent?: string;
      assistantMessage: { content: string };
    };
    assert.equal(orderChatRes.orderInfo?.orderId, "ORD-1001");
    assert.equal(orderChatRes.subIntent, "courier_delay");
    assert.match(orderChatRes.assistantMessage.content, /ORD-1001/);

    const overviewForRollback = (await (
      await fetch(`${baseUrl}/api/overview`)
    ).json()) as { trainingJobs: Array<{ id: string; jobType: string }> };
    const firstEvalJob = overviewForRollback.trainingJobs.find(
      (j) => j.jobType === "intent_eval"
    );
    assert.ok(firstEvalJob);
    const rollbackRes = await fetch(`${baseUrl}/api/models/rollback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId: firstEvalJob.id }),
    });
    assert.equal(rollbackRes.status, 200);

    const metricsRes = (await (
      await fetch(`${baseUrl}/api/metrics`)
    ).json()) as { chatRequests: number; auditLogs: unknown[] };
    assert.ok(metricsRes.chatRequests >= 2);
    assert.ok(metricsRes.auditLogs.length >= 2);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    rmSync(tempDir, { recursive: true, force: true });
  }
});
