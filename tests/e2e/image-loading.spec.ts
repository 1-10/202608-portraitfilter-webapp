import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { FIXTURES } from "./fixturesPath";

for (const [label, filename] of [
  ["JPEG", "portrait.jpg"],
  ["PNG", "portrait.png"],
  ["WebP", "portrait.webp"],
] as const) {
  test(`loads a ${label} image via the file picker`, async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, filename));
    await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
    await expect(page.getByRole("radiogroup", { name: "フィルターを選択" })).toBeVisible();
  });
}

test("loads an image via drag and drop", async ({ page }) => {
  await page.goto("/");

  const filePath = path.join(FIXTURES, "portrait.jpg");

  // Simulate a drop by dispatching a DataTransfer built from the fixture file.
  const buffer = readFileSync(filePath).toString("base64");
  await page.evaluate(
    ({ base64, selector }) => {
      // Decode base64 locally (no fetch/network call — the page's CSP disallows
      // fetching data: URIs anyway, matching the app's own no-network policy).
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const file = new File([bytes], "portrait.jpg", { type: "image/jpeg" });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);

      const target = document.querySelector(selector) as HTMLElement;
      const dropEvent = new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer });
      target.dispatchEvent(dropEvent);
    },
    { base64: buffer, selector: "[class*='dropzone']" },
  );

  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible({ timeout: 10_000 });
});

test("shows a reset-to-idle error message for an unsupported file", async ({ page }) => {
  await page.goto("/");
  const filePath = path.join(FIXTURES, "portrait.jpg");
  const buffer = readFileSync(filePath);
  const input = page.getByLabel("画像ファイルを選択");
  await input.setInputFiles({ name: "portrait.gif", mimeType: "image/gif", buffer });
  await expect(page.getByRole("alert")).toContainText("対応していない画像形式");
});
