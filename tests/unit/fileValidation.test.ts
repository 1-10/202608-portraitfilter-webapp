import { describe, expect, it } from "vitest";
import {
  MAX_DECODED_LONG_EDGE,
  MAX_FILE_SIZE_BYTES,
  validateDecodedDimensions,
  validateImageFile,
} from "../../src/utils/fileValidation";

function makeFile(type: string, sizeBytes: number): File {
  return new File([new Uint8Array(sizeBytes)], "test.bin", { type });
}

describe("validateImageFile", () => {
  it("accepts JPEG, PNG, and WebP within the size limit", () => {
    expect(validateImageFile(makeFile("image/jpeg", 1024)).ok).toBe(true);
    expect(validateImageFile(makeFile("image/png", 1024)).ok).toBe(true);
    expect(validateImageFile(makeFile("image/webp", 1024)).ok).toBe(true);
  });

  it("rejects unsupported formats such as HEIC or GIF", () => {
    const result = validateImageFile(makeFile("image/heic", 1024));
    expect(result).toEqual({ ok: false, reason: "unsupported-format" });
  });

  it("rejects files larger than the configured limit", () => {
    const result = validateImageFile(makeFile("image/png", MAX_FILE_SIZE_BYTES + 1));
    expect(result).toEqual({ ok: false, reason: "too-large" });
  });
});

describe("validateDecodedDimensions", () => {
  it("accepts dimensions within the limit", () => {
    expect(validateDecodedDimensions(1920, 1080).ok).toBe(true);
  });

  it("rejects a decoded image whose long edge exceeds the limit", () => {
    const result = validateDecodedDimensions(MAX_DECODED_LONG_EDGE + 1, 100);
    expect(result).toEqual({ ok: false, reason: "too-large" });
  });
});
