import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { FIXTURES } from "./fixturesPath";
import { bufferContainsAscii, getImageDimensions } from "./utils";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, "portrait.jpg"));
  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
  await page.getByRole("radio", { name: "ロトスコープ" }).click();
});

const FORMATS: { label: string; ext: string }[] = [
  { label: "JPEG", ext: "jpg" },
  { label: "PNG", ext: "png" },
  { label: "WebP", ext: "webp" },
];

for (const { label, ext } of FORMATS) {
  test(`downloads the filtered image as ${label}`, async ({ page }) => {
    await page.getByRole("button", { name: "ダウンロード" }).click();
    await page.getByRole("radio", { name: label }).check();
    await page.getByRole("radio", { name: "長辺1280px" }).check();

    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "ダウンロード" }).nth(-1).click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(new RegExp(`^portrait-rotoscope-\\d{8}-\\d{6}\\.${ext}$`));

    const savedPath = path.join(test.info().outputDir, download.suggestedFilename());
    await download.saveAs(savedPath);

    const dims = await getImageDimensions(page, savedPath);
    expect(Math.max(dims.width, dims.height)).toBe(1280);

    if (ext === "jpg") {
      const buffer = readFileSync(savedPath);
      expect(bufferContainsAscii(buffer, "Exif")).toBe(false);
    }
  });
}

test("resolves 'original size' to the source image's dimensions", async ({ page }) => {
  await page.getByRole("button", { name: "ダウンロード" }).click();
  await expect(page.getByText(/出力サイズ：2400 × 1800 px/)).toBeVisible();
});
