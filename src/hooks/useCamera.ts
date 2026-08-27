import { useCallback, useEffect, useRef, useState } from "react";
import type { AppError } from "../types";

export type CameraStatus = "idle" | "starting" | "active" | "error";
export type FacingMode = "user" | "environment";

function mapCameraError(err: unknown): AppError {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return {
        kind: "camera-permission-denied",
        message: "カメラの使用が許可されませんでした。ブラウザの設定でカメラへのアクセスを許可してください。",
      };
    case "NotFoundError":
    case "DevicesNotFoundError":
      return {
        kind: "camera-not-found",
        message: "カメラが見つかりませんでした。カメラが接続されているか確認してください。",
      };
    case "NotReadableError":
    case "TrackStartError":
      return {
        kind: "camera-unknown",
        message: "カメラを起動できませんでした。他のアプリがカメラを使用している可能性があります。",
      };
    default:
      return {
        kind: "camera-unknown",
        message: "カメラの起動中に不明なエラーが発生しました。",
        detail: err instanceof Error ? err.message : String(err),
      };
  }
}

export type CapturedFrame = { bitmap: ImageBitmap; width: number; height: number };

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<AppError | null>(null);
  const [facingMode, setFacingMode] = useState<FacingMode>("user");
  const [canSwitch, setCanSwitch] = useState(false);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async (mode: FacingMode = "user") => {
    if (!window.isSecureContext) {
      setError({
        kind: "camera-insecure-context",
        message: "カメラを使用するにはHTTPS接続またはlocalhostでアクセスしてください。",
      });
      setStatus("error");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError({ kind: "camera-unknown", message: "このブラウザはカメラ機能に対応していません。" });
      setStatus("error");
      return;
    }

    setStatus("starting");
    setError(null);
    streamRef.current?.getTracks().forEach((track) => track.stop());

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode },
        audio: false,
      });
      streamRef.current = stream;
      setFacingMode(mode);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setStatus("active");

      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setCanSwitch(videoInputs.length > 1);
      } catch {
        setCanSwitch(false);
      }
    } catch (err) {
      setStatus("error");
      setError(mapCameraError(err));
    }
  }, []);

  const switchCamera = useCallback(() => {
    const next: FacingMode = facingMode === "user" ? "environment" : "user";
    return start(next);
  }, [facingMode, start]);

  const capture = useCallback(async (): Promise<CapturedFrame | null> => {
    const video = videoRef.current;
    if (!video || status !== "active") return null;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const bitmap = await createImageBitmap(video);
    return { bitmap, width, height };
  }, [status]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  return { videoRef, status, error, facingMode, canSwitch, start, stop, switchCamera, capture };
}
