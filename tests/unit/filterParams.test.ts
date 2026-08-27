import { describe, expect, it } from "vitest";
import { defaultParamValues, FILTERS, normalizeParamValue } from "../../src/filters/definitions";

describe("normalizeParamValue", () => {
  it("passes through values already within range", () => {
    expect(normalizeParamValue(0, 100, 42)).toBe(42);
  });

  it("clamps values above the max", () => {
    expect(normalizeParamValue(0, 100, 150)).toBe(100);
  });

  it("clamps values below the min", () => {
    expect(normalizeParamValue(0, 100, -20)).toBe(0);
  });

  it("falls back to min for non-finite input", () => {
    expect(normalizeParamValue(3, 10, NaN)).toBe(3);
    expect(normalizeParamValue(3, 10, Infinity)).toBe(3);
  });
});

describe("defaultParamValues", () => {
  it("returns a value for every declared parameter, within range", () => {
    for (const filter of FILTERS) {
      const values = defaultParamValues(filter);
      for (const param of filter.parameters) {
        expect(values[param.id]).toBe(param.defaultValue);
        expect(values[param.id]).toBeGreaterThanOrEqual(param.min);
        expect(values[param.id]).toBeLessThanOrEqual(param.max);
      }
    }
  });
});

describe("FILTERS", () => {
  it("includes exactly the 7 filters with unique ids", () => {
    expect(FILTERS).toHaveLength(7);
    const ids = FILTERS.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("gives every filter at least one shader pass", () => {
    for (const filter of FILTERS) {
      expect(filter.passes.length).toBeGreaterThan(0);
    }
  });

  it("keeps every preset's values within its parameters' declared ranges and known ids", () => {
    for (const filter of FILTERS) {
      const paramById = new Map(filter.parameters.map((p) => [p.id, p]));
      for (const preset of filter.presets ?? []) {
        for (const [paramId, value] of Object.entries(preset.values)) {
          const param = paramById.get(paramId);
          expect(param, `${filter.id}/${preset.id} references unknown parameter "${paramId}"`).toBeDefined();
          expect(value).toBeGreaterThanOrEqual(param!.min);
          expect(value).toBeLessThanOrEqual(param!.max);
        }
      }
    }
  });

  it("gives every extraInputs id a defined earlier pass (or 'original') within the same filter", () => {
    for (const filter of FILTERS) {
      const seenIds = new Set<string>(["original"]);
      for (const pass of filter.passes) {
        for (const inputId of pass.extraInputs ?? []) {
          expect(seenIds.has(inputId), `${filter.id}/${pass.id} references undefined input "${inputId}"`).toBe(true);
        }
        seenIds.add(pass.id);
      }
    }
  });
});
