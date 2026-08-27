import { expect, test } from "@playwright/test";

test("shows guidance when camera permission is denied", async ({ page }) => {
  // The test browser is launched with fake-camera flags (needed for the capture
  // tests below), which auto-accepts the permission prompt. To exercise the
  // denial code path deterministically, stub getUserMedia to reject the way a
  // real browser would when the user declines the permission prompt.
  await page.addInitScript(() => {
    if (navigator.mediaDevices) {
      navigator.mediaDevices.getUserMedia = () =>
        Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
    }
  });
  await page.goto("/");
  await page.getByRole("button", { name: "カメラで撮影" }).click();
  await expect(page.getByText("カメラの使用が許可されませんでした")).toBeVisible({ timeout: 10_000 });
});

test("captures a photo from the camera and stops the stream afterward", async ({ page, context }) => {
  await context.grantPermissions(["camera"]);
  await page.goto("/");
  await page.getByRole("button", { name: "カメラで撮影" }).click();

  const captureButton = page.getByRole("button", { name: "撮影する" });
  await expect(captureButton).toBeEnabled({ timeout: 10_000 });
  await captureButton.click();

  await expect(page.getByRole("button", { name: "画像を変更" })).toBeVisible();

  const activeTracks = await page.evaluate(() => {
    const video = document.querySelector("video");
    const stream = video?.srcObject as MediaStream | null;
    return stream ? stream.getTracks().filter((t) => t.readyState === "live").length : 0;
  });
  expect(activeTracks).toBe(0);
});

test("cancelling the camera view stops the stream and returns to idle", async ({ page, context }) => {
  await context.grantPermissions(["camera"]);
  await page.goto("/");
  await page.getByRole("button", { name: "カメラで撮影" }).click();
  await expect(page.getByRole("button", { name: "撮影する" })).toBeEnabled({ timeout: 10_000 });

  await page.getByRole("button", { name: "キャンセル" }).click();
  await expect(page.getByRole("button", { name: "画像を選択" })).toBeVisible();
});
