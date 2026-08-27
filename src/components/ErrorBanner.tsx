import type { AppError } from "../types";
import styles from "./ErrorBanner.module.css";

type Props = {
  error: AppError;
  onDismiss?: () => void;
  onRetry?: () => void;
};

export default function ErrorBanner({ error, onDismiss, onRetry }: Props) {
  return (
    <div className={styles.banner} role="alert">
      <span className={styles.icon} aria-hidden="true">
        ⚠️
      </span>
      <p className={styles.message}>{error.message}</p>
      <div className={styles.actions}>
        {onRetry && (
          <button type="button" className={styles.actionButton} onClick={onRetry}>
            再試行
          </button>
        )}
        {onDismiss && (
          <button type="button" className={styles.dismissButton} onClick={onDismiss} aria-label="閉じる">
            ×
          </button>
        )}
      </div>
    </div>
  );
}
