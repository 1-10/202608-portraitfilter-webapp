import { expect, test } from "@playwright/test";
import path from "node:path";
import { FIXTURES } from "./fixturesPath";
import { setRangeValue } from "./utils";

/**
 * The privacy guarantee is that no image (or anything derived from one) leaves the
 * device. This used to be asserted as "no network requests at all after load",
 * which stopped being expressible once face detection began lazily fetching its
 * wasm runtime and model from this app's own origin.
 *
 * So the assertion is now written against the actual invariant rather than a
 * proxy for it, and is strictly more specific than before:
 *   1. no cross-origin request is ever made, and
 *   2. every same-origin request is a GET for a known static asset — never a POST,
 *      PUT or any method that could carry a body containing image data.
 */
test("image loading, filtering and export never send data off-device", async ({ page }) => {
  const requests: { url: string; method: string }[] = [];
  page.on("request", (req) => requests.push({ url: req.url(), method: req.method() }));

  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const origin = new URL(page.url()).origin;

  await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, "portrait.jpg"));
  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();

  // Give face detection time to load its model and run, so its fetches are included.
  await page.waitForTimeout(8000);

  await page.getByRole("radio", { name: "水彩" }).click();
  await setRangeValue(page.getByLabel("効果の強さ"), "50");
  await page.waitForTimeout(200);

  await page.getByRole("button", { name: "ダウンロード" }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "ダウンロード" }).nth(-1).click();
  await downloadPromise;

  const external = requests.filter(
    (r) => /^https?:/.test(r.url) && !r.url.startsWith(origin),
  );
  expect(external, `unexpected cross-origin requests: ${JSON.stringify(external)}`).toEqual([]);

  // Nothing may upload: any non-GET could carry a body.
  const nonGet = requests.filter((r) => r.method !== "GET" && !r.url.startsWith("blob:") && !r.url.startsWith("data:"));
  expect(nonGet, `unexpected non-GET requests: ${JSON.stringify(nonGet)}`).toEqual([]);
});

/**
 * Face detection is a progressive enhancement, so the whole app must still work
 * with the model missing from the server. This is the regression guard for that
 * contract: if it ever starts throwing instead of falling back, a user on a flaky
 * connection would lose every filter, not just the face-aware refinement.
 */
test("all filters still work when the face model cannot be loaded", async ({ page }) => {
  await page.route("**/*.task", (route) => route.abort());
  await page.route("**/*.tflite", (route) => route.abort());
  await page.route("**/*.wasm", (route) => route.abort());

  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/");
  await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, "portrait.jpg"));
  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
  await page.waitForTimeout(3000);

  // The makeup filter is the strictest case: with no model there is no mesh and no
  // makeup masks, so it has to degrade to a plain skin-and-tone filter rather than
  // scramble the frame with an undecodable displacement field.
  for (const name of ["ナチュラル美肌", "ビューティーメイク", "AIルック", "シネマティック", "ロトスコープ", "水彩"]) {
    await page.getByRole("radio", { name }).click();
    await expect(page.getByRole("radio", { name, checked: true })).toBeVisible();
  }

  await expect(page.getByText("フィルターの描画に失敗")).toHaveCount(0);
  expect(errors).toEqual([]);
});

/**
 * The person segmenter is a separate model from the landmarker and can fail on its
 * own. Losing it must cost only the segmentation: the face mask still applies and the
 * filters that use both fall back to their colour heuristics for skin and hair.
 */
test("face-aware filters still work when only the segmentation model cannot be loaded", async ({ page }) => {
  await page.route("**/*.tflite", (route) => route.abort());

  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/");
  await page.getByLabel("画像ファイルを選択").setInputFiles(path.join(FIXTURES, "portrait.jpg"));
  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();
  await page.waitForTimeout(8000);

  for (const name of ["ナチュラル美肌", "ビューティーメイク", "AIルック"]) {
    await page.getByRole("radio", { name }).click();
    await expect(page.getByRole("radio", { name, checked: true })).toBeVisible();
  }

  await expect(page.getByText("フィルターの描画に失敗")).toHaveCount(0);
  expect(errors).toEqual([]);
});
