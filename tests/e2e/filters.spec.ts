import { expect, test } from "@playwright/test";
import path from "node:path";
import { FIXTURES } from "./fixturesPath";
import { setRangeValue } from "./utils";

const FILTER_NAMES = ["オリジナル", "ナチュラル美肌", "シネマティック", "ロトスコープ", "水彩", "コミック", "デュオトーン"];

async function loadImage(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, "portrait.jpg"));
  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
}

test("all 7 filters can be selected without error", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await loadImage(page);

  for (const name of FILTER_NAMES) {
    await page.getByRole("radio", { name }).click();
    await expect(page.getByRole("radio", { name, checked: true })).toBeVisible();
  }

  expect(errors).toEqual([]);
});

test("adjusting the strength slider updates the preview canvas", async ({ page }) => {
  await loadImage(page);
  await page.getByRole("radio", { name: "ロトスコープ" }).click();
  await page.waitForTimeout(200);

  // The WebGL canvas is created with preserveDrawingBuffer:false (the app never
  // reads it back itself), so toDataURL()/getImageData() cannot be trusted from
  // the test side either — the buffer may already be cleared by the time the
  // Node-side call round-trips back. A screenshot instead captures whatever was
  // actually composited on screen, which is what we want to verify here.
  const canvas = page.locator("canvas").first();
  const before = await canvas.screenshot();

  await setRangeValue(page.getByLabel("効果の強さ"), "20");
  await page.waitForTimeout(200);

  const after = await canvas.screenshot();
  expect(after.equals(before)).toBe(false);
});

test("compare-original shows the original image while held, and reset restores defaults", async ({ page }) => {
  await loadImage(page);
  await page.getByRole("radio", { name: "デュオトーン" }).click();
  await page.waitForTimeout(200);

  const compareButton = page.getByRole("button", { name: "押している間だけ元画像と比較する" });
  const compareBadge = page.getByTestId("compare-badge");
  const box = await compareButton.boundingBox();
  if (!box) throw new Error("compare button not found");
  const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

  await page.mouse.move(center.x, center.y);
  await page.mouse.down();
  await expect(compareBadge).toBeVisible();
  await page.mouse.up();
  await expect(compareBadge).toHaveCount(0);

  await page.getByRole("button", { name: "リセット" }).click();
  await expect(page.getByRole("radio", { name: "オリジナル", checked: true })).toBeVisible();
});
