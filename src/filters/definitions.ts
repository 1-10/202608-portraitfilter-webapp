import type { FilterDefinition, FilterParamValues } from "../types";
import { aiLookFilter } from "./aiLook";
import { beautyMakeupFilter } from "./beautyMakeup";
import { cinematicFilter } from "./cinematic";
import { comicFilter } from "./comic";
import { duotoneFilter } from "./duotone";
import { originalFilter } from "./original";
import { rotoscopeFilter } from "./rotoscope";
import { skinBeautyFilter } from "./skinBeauty";
import { watercolorFilter } from "./watercolor";

export const FILTERS: FilterDefinition[] = [
  originalFilter,
  skinBeautyFilter,
  beautyMakeupFilter,
  aiLookFilter,
  cinematicFilter,
  rotoscopeFilter,
  watercolorFilter,
  comicFilter,
  duotoneFilter,
];

export const DEFAULT_FILTER_ID = originalFilter.id;

export function getFilterById(id: string): FilterDefinition {
  const found = FILTERS.find((f) => f.id === id);
  return found ?? originalFilter;
}

/** Returns default parameter values for a filter, clamped to each parameter's declared range. */
export function defaultParamValues(filter: FilterDefinition): FilterParamValues {
  const values: FilterParamValues = {};
  for (const param of filter.parameters) {
    values[param.id] = normalizeParamValue(param.min, param.max, param.defaultValue);
  }
  return values;
}

/** Clamps a parameter value to its declared [min, max] range. Non-finite input falls back to min. */
export function normalizeParamValue(min: number, max: number, value: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
