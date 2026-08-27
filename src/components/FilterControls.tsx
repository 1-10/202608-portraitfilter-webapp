import type { FilterDefinition, FilterParamValues } from "../types";
import styles from "./FilterControls.module.css";

type Props = {
  filter: FilterDefinition;
  strength: number;
  paramValues: FilterParamValues;
  onStrengthChange: (value: number) => void;
  onParamChange: (paramId: string, value: number) => void;
  onApplyPreset: (values: FilterParamValues) => void;
  onReset: () => void;
};

export default function FilterControls({
  filter,
  strength,
  paramValues,
  onStrengthChange,
  onParamChange,
  onApplyPreset,
  onReset,
}: Props) {
  const hasStrength = filter.hasStrength !== false;

  if (!hasStrength && filter.parameters.length === 0) {
    return (
      <div className={styles.wrapper}>
        <p className={styles.emptyNote}>このフィルターに調整項目はありません。</p>
        <button type="button" className={styles.resetButton} onClick={onReset}>
          リセット
        </button>
      </div>
    );
  }

  return (
    <div className={styles.wrapper}>
      {filter.presets && filter.presets.length > 0 && (
        <div className={styles.row}>
          <div className={styles.rowHeader}>
            <span className={styles.label}>プリセット</span>
          </div>
          <div className={styles.optionGroup} role="group" aria-label="プリセット">
            {filter.presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={styles.optionButton}
                onClick={() => onApplyPreset(preset.values)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {hasStrength && (
        <div className={styles.row}>
          <div className={styles.rowHeader}>
            <label htmlFor="strength-slider" className={styles.label}>
              効果の強さ
            </label>
            <span className={styles.value}>{strength}</span>
          </div>
          <input
            id="strength-slider"
            type="range"
            min={0}
            max={100}
            step={1}
            value={strength}
            onChange={(e) => onStrengthChange(Number(e.target.value))}
            aria-label="効果の強さ"
            className={styles.slider}
          />
        </div>
      )}

      {filter.parameters.map((param) => {
        const value = paramValues[param.id] ?? param.defaultValue;
        if (param.options) {
          return (
            <div className={styles.row} key={param.id}>
              <div className={styles.rowHeader}>
                <span className={styles.label}>{param.label}</span>
              </div>
              <div className={styles.optionGroup} role="radiogroup" aria-label={param.label}>
                {param.options.map((optionLabel, index) => (
                  <button
                    key={optionLabel}
                    type="button"
                    role="radio"
                    aria-checked={value === index}
                    className={value === index ? `${styles.optionButton} ${styles.optionButtonSelected}` : styles.optionButton}
                    onClick={() => onParamChange(param.id, index)}
                  >
                    {optionLabel}
                  </button>
                ))}
              </div>
            </div>
          );
        }
        return (
          <div className={styles.row} key={param.id}>
            <div className={styles.rowHeader}>
              <label htmlFor={`param-${param.id}`} className={styles.label}>
                {param.label}
              </label>
              <span className={styles.value}>{value}</span>
            </div>
            <input
              id={`param-${param.id}`}
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={value}
              onChange={(e) => onParamChange(param.id, Number(e.target.value))}
              aria-label={param.label}
              className={styles.slider}
            />
          </div>
        );
      })}

      <button type="button" className={styles.resetButton} onClick={onReset}>
        リセット
      </button>
    </div>
  );
}
