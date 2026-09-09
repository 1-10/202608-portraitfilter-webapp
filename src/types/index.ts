export type FilterParameter = {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  /** When set, the parameter is a discrete choice rendered as labeled options instead of a plain slider. */
  options?: string[];
};

export type ShaderPassDefinition = {
  id: string;
  /**
   * Human-readable name for the dev-only debug pass picker. Optional; the picker
   * falls back to the id.
   *
   * Declared HERE rather than in a lookup table beside the picker: a table keyed by
   * pass id is a copy of information that lives in this file, and it goes stale
   * silently the moment a pass is renamed or added.
   */
  label?: string;
  /** Fragment shader GLSL source (ES3, #version 300 es). */
  fragmentSource: string;
  /** Scale factor applied to the working resolution for this pass's output framebuffer (1 = full working res). */
  outputScale?: number;
  /**
   * Hard cap on this pass's output long edge, in pixels, applied after outputScale.
   *
   * Radii inside shaders are expressed in texels, so without a cap a 3px outline
   * authored against the 1280px preview renders as a 3px hairline in a 4096px
   * export — proportionally 3x thinner than what the user approved. Capping the
   * expensive stylization passes makes the export a faithful upscale of the
   * approved look, and keeps cost bounded. Leave unset for cheap passes that
   * should genuinely run at full output resolution (final composites, line work).
   */
  maxOutputLongEdge?: number;
  /**
   * IDs of earlier passes in the same filter (or the literal "original") whose
   * output should be bound as extra sampler inputs named `uInput_<id>`.
   */
  extraInputs?: string[];
};

export type FilterPreset = {
  id: string;
  label: string;
  /** Parameter id -> value. Unlisted parameters keep their defaultValue. */
  values: FilterParamValues;
};

export type FilterDefinition = {
  id: string;
  name: string;
  description: string;
  parameters: FilterParameter[];
  passes: ShaderPassDefinition[];
  /** Whether the common "効果の強さ" slider applies. Defaults to true. */
  hasStrength?: boolean;
  /** Optional named bundles of parameter values, shown as quick-select buttons. */
  presets?: FilterPreset[];
};

export type FilterParamValues = Record<string, number>;

export type OutputFormat = "image/jpeg" | "image/png" | "image/webp";

export type ExportSizePreset = "original" | "1920" | "1280";

export type AppState =
  | "idle"
  | "loading-image"
  | "camera-preview"
  | "processing-preview"
  | "ready"
  | "exporting"
  | "error";

export type AppErrorKind =
  | "unsupported-format"
  | "decode-failed"
  | "too-large"
  | "webgl-unsupported"
  | "shader-compile-failed"
  | "camera-permission-denied"
  | "camera-not-found"
  | "camera-insecure-context"
  | "camera-unknown"
  | "export-failed"
  | "context-lost"
  | "unknown";

export type AppError = {
  kind: AppErrorKind;
  message: string;
  detail?: string;
};

export type LoadedImage = {
  bitmap: ImageBitmap;
  width: number;
  height: number;
  sourceFileName: string | null;
};
