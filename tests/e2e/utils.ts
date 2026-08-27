import type { Locator, Page } from "@playwright/test";
import { readFileSync } from "node:fs";

/** Decodes an image file's pixel dimensions using the browser's own image decoder. */
export async function getImageDimensions(page: Page, filePath: string): Promise<{ width: number; height: number }> {
  const buffer = readFileSync(filePath);
  const base64 = buffer.toString("base64");
  const ext = filePath.split(".").pop() ?? "png";
  const mime = ext === "jpg" ? "jpeg" : ext;
  return page.evaluate(
    ({ base64, mime }) => {
      return new Promise<{ width: number; height: number }>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = () => reject(new Error("image decode failed"));
        img.src = `data:image/${mime};base64,${base64}`;
      });
    },
    { base64, mime },
  );
}

export function bufferContainsAscii(buffer: Buffer, needle: string): boolean {
  return buffer.includes(Buffer.from(needle, "ascii"));
}

/**
 * Sets a range input's value through React's native property setter (bypassing
 * the instance-level setter React installs to track controlled inputs) and
 * dispatches real input/change events, matching how a user drag would behave.
 */
export async function setRangeValue(locator: Locator, value: string): Promise<void> {
  await locator.evaluate((el, val) => {
    const proto = Object.getPrototypeOf(el) as object;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    setter?.call(el, val);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
}
