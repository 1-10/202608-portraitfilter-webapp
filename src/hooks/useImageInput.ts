import { useCallback, useState } from "react";
import type { AppError, LoadedImage } from "../types";
import { decodeImageFile, ImageDecodeError } from "../utils/imageDecode";

export function useImageInput() {
  const [error, setError] = useState<AppError | null>(null);

  const loadFile = useCallback(async (file: File): Promise<LoadedImage | null> => {
    setError(null);
    try {
      return await decodeImageFile(file);
    } catch (err) {
      if (err instanceof ImageDecodeError) {
        setError(err.appError);
      } else {
        setError({
          kind: "decode-failed",
          message: "画像の読み込みに失敗しました。",
          detail: err instanceof Error ? err.message : String(err),
        });
      }
      return null;
    }
  }, []);

  return { loadFile, error, clearError: () => setError(null) };
}
