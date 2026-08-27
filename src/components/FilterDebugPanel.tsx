import { useEffect, useState } from "react";
import { defaultParamValues } from "../filters/definitions";
import type { FilterDefinition, FilterParamValues } from "../types";
import styles from "./FilterDebugPanel.module.css";

type Props = {
  filter: FilterDefinition;
  paramValues: FilterParamValues;
  strength: number;
  renderNow: (filter: FilterDefinition, params: FilterParamValues, strengthPercent: number, debugPassId?: string) => void;
  generateThumbnail: (
    filter: FilterDefinition,
    size?: number,
    paramsOverride?: FilterParamValues,
    strengthOverride?: number,
  ) => string | null;
  originalFilter: FilterDefinition;
  faceMaskActive: boolean;
  faceScale: number;
};

const PASS_LABELS: Record<string, string> = {
  structure: "structure (色面)",
  region: "region (肌/水分)",
  pigmentWash: "pigmentWash (顔料)",
  bleedH1: "bleedH1",
  bleedV1: "bleedV1",
  bleedH2: "bleedH2",
  bleedV2: "bleedV2",
  edgeDeposit: "edgeDeposit (沈着)",
  detailRestore: "detailRestore (輪郭)",
  paperInteract: "paperInteract (紙)",
  finalComposite: "finalComposite (最終)",
};

/**
 * Dev-only debug view for the watercolor pipeline (spec §11): lets you pick any
 * intermediate pass to view its raw output on the live canvas, and renders a
 * side-by-side thumbnail comparison of the original image against the 3
 * presets. Tree-shaken out of production builds since callers gate this
 * behind `import.meta.env.DEV`.
 */
export default function FilterDebugPanel({
  filter,
  paramValues,
  strength,
  renderNow,
  generateThumbnail,
  originalFilter,
  faceMaskActive,
  faceScale,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const [debugPassId, setDebugPassId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{ label: string; dataUrl: string | null }[] | null>(null);

  useEffect(() => {
    if (debugPassId) {
      renderNow(filter, paramValues, strength, debugPassId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debugPassId, paramValues, strength]);

  const selectPass = (passId: string | null) => {
    setDebugPassId(passId);
    if (!passId) renderNow(filter, paramValues, strength);
  };

  const runComparison = () => {
    const shots: { label: string; dataUrl: string | null }[] = [
      { label: "元画像", dataUrl: generateThumbnail(originalFilter, 160) },
    ];
    for (const preset of filter.presets ?? []) {
      const values = { ...defaultParamValues(filter), ...preset.values };
      shots.push({ label: preset.label, dataUrl: generateThumbnail(filter, 160, values, 100) });
    }
    setComparison(shots);
    // Restore whatever the live canvas was showing before the comparison run.
    renderNow(filter, paramValues, strength, debugPassId ?? undefined);
  };

  if (!expanded) {
    return (
      <button type="button" className={styles.toggleButton} onClick={() => setExpanded(true)}>
        🐛 デバッグ表示 (dev)
      </button>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>
          🐛 パイプライン デバッグ (dev only) —{" "}
          {faceMaskActive ? `顔検出 OK (faceScale ${faceScale.toFixed(3)})` : "顔マスクなし（ヒューリスティックで動作中）"}
        </span>
        <button type="button" className={styles.toggleButton} onClick={() => setExpanded(false)}>
          閉じる
        </button>
      </div>

      <div className={styles.section}>
        <p className={styles.sectionLabel}>中間パスを表示（プレビューを上書き）</p>
        <div className={styles.passRow}>
          <button
            type="button"
            className={debugPassId === null ? `${styles.passButton} ${styles.passButtonActive}` : styles.passButton}
            onClick={() => selectPass(null)}
          >
            最終結果
          </button>
          {filter.passes.map((pass) => (
            <button
              key={pass.id}
              type="button"
              className={debugPassId === pass.id ? `${styles.passButton} ${styles.passButtonActive}` : styles.passButton}
              onClick={() => selectPass(pass.id)}
            >
              {PASS_LABELS[pass.id] ?? pass.id}
            </button>
          ))}
        </div>
        {debugPassId && <p className={styles.hint}>raw RGBA — 一部パスは色ではなく濃度/マスク値をチャンネルへ格納しています</p>}
      </div>

      <div className={styles.section}>
        <p className={styles.sectionLabel}>プリセット比較（元画像 / 3プリセット）</p>
        <button type="button" className={styles.passButton} onClick={runComparison}>
          比較サムネイルを生成
        </button>
        {comparison && (
          <div className={styles.comparisonRow}>
            {comparison.map((shot) => (
              <div key={shot.label} className={styles.comparisonItem}>
                {shot.dataUrl ? <img src={shot.dataUrl} alt={shot.label} className={styles.comparisonImg} /> : <div className={styles.comparisonImg} />}
                <span className={styles.comparisonLabel}>{shot.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
