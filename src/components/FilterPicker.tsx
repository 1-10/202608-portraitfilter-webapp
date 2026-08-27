import type { FilterDefinition } from "../types";
import styles from "./FilterPicker.module.css";

type Props = {
  filters: FilterDefinition[];
  selectedId: string;
  thumbnails: Record<string, string | null>;
  onSelect: (filterId: string) => void;
};

export default function FilterPicker({ filters, selectedId, thumbnails, onSelect }: Props) {
  return (
    <div className={styles.wrapper} role="radiogroup" aria-label="フィルターを選択">
      <ul className={styles.list}>
        {filters.map((filter) => {
          const selected = filter.id === selectedId;
          const thumbnail = thumbnails[filter.id];
          return (
            <li key={filter.id}>
              <button
                type="button"
                role="radio"
                aria-checked={selected}
                className={selected ? `${styles.item} ${styles.itemSelected}` : styles.item}
                onClick={() => onSelect(filter.id)}
              >
                <span className={styles.thumbWrap}>
                  {thumbnail ? (
                    <img src={thumbnail} alt="" className={styles.thumb} />
                  ) : (
                    <span className={styles.thumbPlaceholder} aria-hidden="true" />
                  )}
                </span>
                <span className={styles.label}>{filter.name}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
