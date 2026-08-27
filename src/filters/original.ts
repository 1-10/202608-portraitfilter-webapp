import { buildFragmentShader } from "../rendering/shaders/glslCommon";
import type { FilterDefinition } from "../types";

const passthrough = buildFragmentShader({
  body: "  fragColor = texture(uSource, vUv);",
});

export const originalFilter: FilterDefinition = {
  id: "original",
  name: "オリジナル",
  description: "無加工のオリジナル画像です。比較やリセットの基準になります。",
  parameters: [],
  hasStrength: false,
  passes: [{ id: "identity", fragmentSource: passthrough }],
};
