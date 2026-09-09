import { expect, test } from "@playwright/test";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";

/**
 * Renders the makeup filter's presets from a real photograph and writes them out
 * for eyeballing, and times a full-resolution export of the heaviest filter.
 *
 * SKIPPED unless `samples/portrait.png` exists. The filter is tuned against a
 * photograph of a real person, and that photograph cannot live in the repository —
 * so this spec names where to put one instead of pretending the tuning was done
 * against the synthetic fixtures, which have no face in them at all.
 *
 * Put any portrait at samples/portrait.png (the directory is gitignored) and run
 * `npx playwright test visual-reference`. Output lands in samples/out/.
 */
const SAMPLE = path.join(process.cwd(), "samples", "portrait.png");
const OUT_DIR = path.join(process.cwd(), "samples", "out");
const PRESETS = ["ほんのり", "ナチュラル", "しっかり"];

test.describe("beauty-makeup against a real portrait", () => {
  test.skip(!existsSync(SAMPLE), "samples/portrait.png not present");
  test.slow();

  test("renders every preset and times a full-resolution export", async ({ page }) => {
    mkdirSync(OUT_DIR, { recursive: true });
    const problems: string[] = [];
    page.on("pageerror", (err) => problems.push(`[pageerror] ${err.message}`));
    page.on("console", (message) => {
      if (message.text().includes("[faceGeometry]")) problems.push(message.text());
    });

    await page.goto("/");
    await page.getByLabel("画像ファイルを選択").setInputFiles(SAMPLE);
    await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
    await page.getByRole("radio", { name: "ビューティーメイク" }).click();

    // Detection is asynchronous and deliberately silent, so there is no event to
    // await. The mesh bake follows it immediately; a fixed wait is enough here
    // because a MISSING bake would still render (as a plain skin filter) and the
    // written-out images would show it.
    await page.waitForTimeout(6000);

    const download = async (label: string, size: string) => {
      await page.getByRole("button", { name: "ダウンロード" }).first().click();
      await page.getByRole("radio", { name: "PNG" }).check();
      await page.getByRole("radio", { name: size }).check();
      const pending = page.waitForEvent("download", { timeout: 180_000 });
      const started = Date.now();
      await page.getByRole("button", { name: "ダウンロード" }).nth(-1).click();
      const file = await pending;
      const elapsed = Date.now() - started;
      await file.saveAs(path.join(OUT_DIR, `${label}.png`));
      await page.getByRole("button", { name: "閉じる" }).click();
      return elapsed;
    };

    for (const preset of PRESETS) {
      await page.getByRole("button", { name: preset, exact: true }).click();
      await page.waitForTimeout(400);
      await download(preset, "長辺1920px");
    }

    await page.getByRole("radio", { name: "オリジナル" }).click();
    await page.waitForTimeout(300);
    await download("original", "長辺1920px");

    // Full-resolution timing, on the preset that runs every pass at full weight.
    await page.getByRole("radio", { name: "ビューティーメイク" }).click();
    await page.getByRole("button", { name: "しっかり", exact: true }).click();
    await page.waitForTimeout(400);
    const fullMs = await download("full-resolution", "元のサイズ");
    console.log(`[timing] full-resolution export: ${(fullMs / 1000).toFixed(2)}s`);

    await expect(page.getByText("フィルターの描画に失敗")).toHaveCount(0);
    expect(problems).toEqual([]);
  });
});
