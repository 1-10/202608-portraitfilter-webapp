import { useEffect } from "react";
import type { CapturedFrame } from "../hooks/useCamera";
import { useCamera } from "../hooks/useCamera";
import styles from "./CameraCapture.module.css";

type Props = {
  onCapture: (frame: CapturedFrame) => void;
  onCancel: () => void;
};

export default function CameraCapture({ onCapture, onCancel }: Props) {
  const { videoRef, status, error, canSwitch, start, stop, switchCamera, capture } = useCamera();

  useEffect(() => {
    void start("user");
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCapture = async () => {
    const frame = await capture();
    if (!frame) return;
    stop();
    onCapture(frame);
  };

  const handleCancel = () => {
    stop();
    onCancel();
  };

  return (
    <div className={styles.wrapper}>
      <div className={styles.videoStage}>
        <video ref={videoRef} className={styles.video} playsInline muted aria-label="カメラのライブプレビュー" />
        {status === "starting" && <p className={styles.overlayMessage}>カメラを起動しています…</p>}
        {status === "error" && error && (
          <div className={styles.overlayMessage}>
            <p>{error.message}</p>
            <button type="button" className={styles.retryButton} onClick={() => start("user")}>
              再試行
            </button>
          </div>
        )}
      </div>

      <div className={styles.controls}>
        <button type="button" className={styles.cancelButton} onClick={handleCancel}>
          キャンセル
        </button>
        <button
          type="button"
          className={styles.captureButton}
          onClick={handleCapture}
          disabled={status !== "active"}
          aria-label="撮影する"
        >
          撮影
        </button>
        {canSwitch ? (
          <button type="button" className={styles.switchButton} onClick={switchCamera} disabled={status !== "active"}>
            カメラ切替
          </button>
        ) : (
          <span className={styles.switchPlaceholder} aria-hidden="true" />
        )}
      </div>
    </div>
  );
}
