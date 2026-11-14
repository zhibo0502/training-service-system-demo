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

    // 3. 触发新实验（oversample + seed=7）
    const expRes = (await (
      await fetch(`${baseUrl}/api/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version: "v1.1.0",
          seed: 7,
          balanceStrategy: "oversample",
        }),
      })
    ).json()) as { datasetVersion: { version: string; totalCount: number } };
    assert.equal(expRes.datasetVersion.version, "v1.1.0");
    assert.equal(expRes.datasetVersion.totalCount, 28);

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

    // 5. 人工客服认领并回复
    const replyRes = await fetch(`${baseUrl}/api/handoff/reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticketId: chatRes.ticket.id,
        replyContent: "您好，人工客服已接手为您处理。",
      }),
    });
    assert.equal(replyRes.status, 200);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    rmSync(tempDir, { recursive: true, force: true });
  }
});
