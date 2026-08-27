import styles from "./Header.module.css";

type Props = {
  hasImage: boolean;
  onChangeImage: () => void;
};

export default function Header({ hasImage, onChangeImage }: Props) {
  return (
    <header className={styles.header}>
      <div className={styles.titleGroup}>
        <span className={styles.logo} aria-hidden="true">
          🎨
        </span>
        <div>
          <h1 className={styles.title}>人物画像フィルター</h1>
          <p className={styles.subtitle}>端末内だけで完結する簡易画風フィルター</p>
        </div>
      </div>
      {hasImage && (
        <button type="button" className={styles.changeButton} onClick={onChangeImage}>
          画像を変更
        </button>
      )}
    </header>
  );
}
