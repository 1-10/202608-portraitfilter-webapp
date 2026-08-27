import { type RefObject } from "react";
import styles from "./ImagePreview.module.css";

type Props = {
  canvasRef: RefObject<HTMLCanvasElement>;
  overlayCanvasRef: RefObject<HTMLCanvasElement>;
  compareHeld: boolean;
  isProcessing: boolean;
  onCompareStart: () => void;
  onCompareEnd: () => void;
};

export default function ImagePreview({
  canvasRef,
  overlayCanvasRef,
  compareHeld,
  isProcessing,
  onCompareStart,
  onCompareEnd,
}: Props) {
  return (
    <div className={styles.stage}>
      <div className={styles.canvasWrap}>
        <canvas ref={canvasRef} className={styles.canvas} />
        <canvas
          ref={overlayCanvasRef}
          className={compareHeld ? `${styles.canvas} ${styles.overlayVisible}` : `${styles.canvas} ${styles.overlayHidden}`}
        />
        {isProcessing && (
          <div className={styles.processingBadge} role="status" aria-live="polite">
            <span className={styles.spinner} aria-hidden="true" />
            処理中…
          </div>
        )}
        {compareHeld && (
          <div className={styles.compareBadge} data-testid="compare-badge">
            オリジナル
          </div>
        )}
      </div>

      <button
        type="button"
        className={styles.compareButton}
        onMouseDown={onCompareStart}
        onMouseUp={onCompareEnd}
        onMouseLeave={onCompareEnd}
        onTouchStart={(e) => {
          e.preventDefault();
          onCompareStart();
        }}
        onTouchEnd={onCompareEnd}
        aria-pressed={compareHeld}
        aria-label="押している間だけ元画像と比較する"
      >
        元画像と比較
      </button>
    </div>
  );
}
