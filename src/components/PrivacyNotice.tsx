import styles from "./PrivacyNotice.module.css";

type Props = {
  compact?: boolean;
};

export default function PrivacyNotice({ compact = false }: Props) {
  return (
    <p className={compact ? styles.compact : styles.full} role="note">
      <span aria-hidden="true">🔒</span> 画像処理はこの端末内だけで行われ、外部へ送信されません。
    </p>
  );
}
