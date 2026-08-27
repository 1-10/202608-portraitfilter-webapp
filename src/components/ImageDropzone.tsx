import { useCallback, useRef, useState, type DragEvent } from "react";
import { ACCEPTED_MIME_TYPES } from "../utils/fileValidation";
import PrivacyNotice from "./PrivacyNotice";
import styles from "./ImageDropzone.module.css";

type Props = {
  onFileSelected: (file: File) => void;
  onCameraRequested: () => void;
  busy: boolean;
};

export default function ImageDropzone({ onFileSelected, onCameraRequested, busy }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected],
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragOver(false);
      handleFiles(event.dataTransfer.files);
    },
    [handleFiles],
  );

  return (
    <div className={styles.wrapper}>
      <div
        className={isDragOver ? `${styles.dropzone} ${styles.dropzoneActive}` : styles.dropzone}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <p className={styles.icon} aria-hidden="true">
          🖼️
        </p>
        <p className={styles.heading}>人物写真をドラッグ＆ドロップ</p>
        <p className={styles.description}>またはボタンから画像を選択・カメラで撮影してください</p>

        <div className={styles.actions}>
          <button type="button" className={styles.primaryButton} onClick={() => inputRef.current?.click()} disabled={busy}>
            画像を選択
          </button>
          <button type="button" className={styles.secondaryButton} onClick={onCameraRequested} disabled={busy}>
            カメラで撮影
          </button>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_MIME_TYPES.join(",")}
          className="visually-hidden"
          aria-label="画像ファイルを選択"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />

        <p className={styles.formats}>対応形式：JPEG / PNG / WebP</p>
      </div>
      <PrivacyNotice />
    </div>
  );
}
