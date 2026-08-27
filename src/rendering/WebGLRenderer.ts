import type { FilterDefinition, FilterParamValues } from "../types";
import { ShaderProgram } from "./ShaderProgram";
import { TextureManager, type PooledTarget } from "./TextureManager";
import { buildFragmentShader, FULLSCREEN_VERT, inputUniform, paramUniform } from "./shaders/glslCommon";

export class WebGLUnsupportedError extends Error {}

export type RenderTargetPixels = {
  data: Uint8ClampedArray<ArrayBuffer>;
  width: number;
  height: number;
};

const BLIT_FRAGMENT = buildFragmentShader({ body: "  fragColor = texture(uSource, vUv);" });

/** TextureManager key for the MediaPipe-derived face mask. */
const FACE_MASK_TEXTURE = "faceMask";

/** Assumed interocular fraction when no face was detected. Mirrors useFilterRenderer. */
const DEFAULT_FACE_SCALE = 0.14;

/**
 * Owns a single WebGL2 context and runs multi-pass filter pipelines against it.
 * Reused for both interactive preview and full-resolution export so shader
 * programs only need to be compiled once per filter.
 */
export class WebGLRenderer {
  readonly gl: WebGL2RenderingContext;
  readonly maxTextureSize: number;
  private readonly canvas: HTMLCanvasElement;
  private readonly textures: TextureManager;
  private readonly programCache = new Map<string, { program: ShaderProgram; source: string }>();
  private readonly emptyVao: WebGLVertexArrayObject;
  private blitProgram: ShaderProgram | null = null;

  private lastImageSource: TexImageSource | null = null;
  private lastImageWidth = 0;
  private lastImageHeight = 0;
  private lastFaceMaskSource: TexImageSource | null = null;
  private faceScale = DEFAULT_FACE_SCALE;

  private contextLost = false;
  private onLost: (() => void) | null = null;
  private onRestored: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      throw new WebGLUnsupportedError("WebGL2に対応していません。");
    }
    this.gl = gl;
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    this.textures = new TextureManager(gl);

    const vao = gl.createVertexArray();
    if (!vao) throw new WebGLUnsupportedError("VAOを作成できませんでした。");
    this.emptyVao = vao;

    canvas.addEventListener("webglcontextlost", this.handleContextLost);
    canvas.addEventListener("webglcontextrestored", this.handleContextRestored);
  }

  setContextEventHandlers(onLost: () => void, onRestored: () => void): void {
    this.onLost = onLost;
    this.onRestored = onRestored;
  }

  isContextLost(): boolean {
    return this.contextLost;
  }

  private handleContextLost = (event: Event): void => {
    event.preventDefault();
    this.contextLost = true;
    this.programCache.clear();
    this.onLost?.();
  };

  private handleContextRestored = (): void => {
    this.contextLost = false;
    this.blitProgram = null;
    if (this.lastImageSource) {
      this.textures.setOriginalImage(this.lastImageSource, this.lastImageWidth, this.lastImageHeight);
    }
    if (this.lastFaceMaskSource) {
      this.textures.setAuxTexture(FACE_MASK_TEXTURE, this.lastFaceMaskSource);
    }
    this.onRestored?.();
  };

  setImage(source: TexImageSource, width: number, height: number): void {
    this.textures.setOriginalImage(source, width, height);
    this.lastImageSource = source;
    this.lastImageWidth = width;
    this.lastImageHeight = height;
  }

  /**
   * Supplies (or clears with `null`) the face-region mask that every pass of every
   * filter can sample as `uFaceMask`. The mask is a small fixed-resolution texture
   * addressed by normalized UV, so it deliberately does NOT need regenerating when
   * the render resolution changes between preview, thumbnail and export.
   */
  setFaceMask(source: TexImageSource | null, faceScale = DEFAULT_FACE_SCALE): void {
    if (source) {
      this.textures.setAuxTexture(FACE_MASK_TEXTURE, source);
      this.lastFaceMaskSource = source;
      this.faceScale = faceScale;
    } else {
      this.textures.clearAuxTexture(FACE_MASK_TEXTURE);
      this.lastFaceMaskSource = null;
      this.faceScale = DEFAULT_FACE_SCALE;
    }
  }

  hasFaceMask(): boolean {
    return this.lastFaceMaskSource !== null;
  }

  private getProgram(key: string, fragmentSource: string): ShaderProgram {
    const cached = this.programCache.get(key);
    // The source is part of the identity, not just the key: during dev hot-reload
    // a pass keeps its id while its GLSL changes, and keying on id alone would
    // silently keep running the stale program.
    if (cached && cached.source === fragmentSource) return cached.program;
    cached?.program.dispose();
    const program = new ShaderProgram(this.gl, FULLSCREEN_VERT, fragmentSource);
    this.programCache.set(key, { program, source: fragmentSource });
    return program;
  }

  private runPipeline(
    filter: FilterDefinition,
    params: FilterParamValues,
    strengthPercent: number,
    width: number,
    height: number,
    debugPassId?: string,
  ): PooledTarget {
    const gl = this.gl;
    const originalTexture = this.textures.getOriginalTexture();
    if (!originalTexture) {
      throw new Error("画像が読み込まれていません。");
    }
    if (filter.passes.length === 0) {
      throw new Error("フィルターにパスが定義されていません。");
    }

    this.textures.beginFrame();
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    const outputsById = new Map<string, PooledTarget>();
    const originalSize = this.textures.getOriginalSize();
    let chainInput: { texture: WebGLTexture; width: number; height: number } = {
      texture: originalTexture,
      width: originalSize.width,
      height: originalSize.height,
    };

    let finalTarget: PooledTarget | null = null;

    for (const pass of filter.passes) {
      const scale = pass.outputScale ?? 1;
      let outW = Math.max(1, Math.round(width * scale));
      let outH = Math.max(1, Math.round(height * scale));
      if (pass.maxOutputLongEdge) {
        const capScale = Math.min(1, pass.maxOutputLongEdge / Math.max(outW, outH));
        outW = Math.max(1, Math.round(outW * capScale));
        outH = Math.max(1, Math.round(outH * capScale));
      }
      const target = this.textures.acquireTarget(`${filter.id}:${pass.id}`, outW, outH);

      const program = this.getProgram(`${filter.id}:${pass.id}`, pass.fragmentSource);
      program.use();

      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, outW, outH);

      let unit = 0;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, chainInput.texture);
      program.setInt("uSource", unit);
      unit += 1;

      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, originalTexture);
      program.setInt("uOriginal", unit);
      unit += 1;

      // The face mask occupies a fixed unit for every pass of every filter, so any
      // shader can consult it without the filter having to declare it. Always bind
      // something — an unbound sampler2D is undefined behaviour.
      const faceMaskTexture = this.textures.getAuxTexture(FACE_MASK_TEXTURE);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, faceMaskTexture ?? this.textures.getPlaceholderMaskTexture());
      program.setInt("uFaceMask", unit);
      program.setFloat("uHasFaceMask", faceMaskTexture ? 1 : 0);
      program.setFloat("uFaceScale", this.faceScale);
      unit += 1;

      for (const inputId of pass.extraInputs ?? []) {
        const source = inputId === "original" ? { texture: originalTexture } : outputsById.get(inputId);
        if (!source) continue;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, source.texture);
        program.setInt(inputUniform(inputId), unit);
        unit += 1;
      }

      program.setVec2("uResolution", outW, outH);
      program.setVec2("uTexelSize", 1 / chainInput.width, 1 / chainInput.height);
      program.setVec2("uOriginalTexelSize", 1 / originalSize.width, 1 / originalSize.height);
      program.setFloat("uStrength", strengthPercent / 100);

      for (const paramDef of filter.parameters) {
        const value = params[paramDef.id] ?? paramDef.defaultValue;
        program.setFloat(paramUniform(paramDef.id), value);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 3);

      outputsById.set(pass.id, target);
      chainInput = target;
      finalTarget = target;
    }

    this.textures.evictStale();
    if (debugPassId) {
      const debugTarget = outputsById.get(debugPassId);
      if (debugTarget) return debugTarget;
    }
    // finalTarget is always set: the loop runs at least once (passes.length > 0 checked above).
    return finalTarget!;
  }

  /**
   * Runs the pipeline and blits the result onto the visible canvas at (width, height).
   * When `debugPassId` names one of the filter's passes, that pass's own output is
   * blitted instead of the final composited result — used by the debug pass viewer.
   */
  renderPreview(
    filter: FilterDefinition,
    params: FilterParamValues,
    strengthPercent: number,
    width: number,
    height: number,
    debugPassId?: string,
  ): void {
    const gl = this.gl;
    const result = this.runPipeline(filter, params, strengthPercent, width, height, debugPassId);

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }

    if (!this.blitProgram) {
      this.blitProgram = new ShaderProgram(this.gl, FULLSCREEN_VERT, BLIT_FRAGMENT);
    }
    this.blitProgram.use();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, result.texture);
    this.blitProgram.setInt("uSource", 0);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Runs the pipeline off-screen and reads back top-down RGBA pixel data. */
  renderToPixels(
    filter: FilterDefinition,
    params: FilterParamValues,
    strengthPercent: number,
    width: number,
    height: number,
  ): RenderTargetPixels {
    const gl = this.gl;
    const result = this.runPipeline(filter, params, strengthPercent, width, height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, result.framebuffer);

    const bottomUp = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);

    // WebGL's readback origin is bottom-left; flip to top-down for canvas/image use.
    const topDown = new Uint8ClampedArray(width * height * 4);
    const rowBytes = width * 4;
    for (let y = 0; y < height; y++) {
      const srcStart = (height - 1 - y) * rowBytes;
      topDown.set(bottomUp.subarray(srcStart, srcStart + rowBytes), y * rowBytes);
    }

    return { data: topDown, width, height };
  }

  dispose(): void {
    this.canvas.removeEventListener("webglcontextlost", this.handleContextLost);
    this.canvas.removeEventListener("webglcontextrestored", this.handleContextRestored);
    for (const cached of this.programCache.values()) cached.program.dispose();
    this.programCache.clear();
    this.blitProgram?.dispose();
    this.textures.disposeAll();
    this.gl.deleteVertexArray(this.emptyVao);
  }
}
