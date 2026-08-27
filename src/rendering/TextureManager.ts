export type PooledTarget = {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
  width: number;
  height: number;
};

/**
 * Manages GPU textures and framebuffers, reusing render-target objects across
 * frames instead of allocating new ones for every render. Entries are keyed by
 * a caller-supplied string (typically `${filterId}:${passId}`) so a given
 * pass's target is kept alive and merely resized when dimensions change.
 */
export class TextureManager {
  private readonly gl: WebGL2RenderingContext;
  private readonly pool = new Map<string, PooledTarget & { lastUsedFrame: number }>();
  private frame = 0;
  private originalTexture: WebGLTexture | null = null;
  private originalWidth = 0;
  private originalHeight = 0;
  private readonly auxTextures = new Map<string, WebGLTexture>();
  private placeholderMaskTexture: WebGLTexture | null = null;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  beginFrame(): void {
    this.frame += 1;
  }

  /** Uploads (or re-uploads) the working-resolution original image as a texture. */
  setOriginalImage(source: TexImageSource, width: number, height: number): WebGLTexture {
    const gl = this.gl;
    if (!this.originalTexture) {
      const tex = gl.createTexture();
      if (!tex) throw new Error("テクスチャを作成できませんでした。");
      this.originalTexture = tex;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.originalTexture);
    // Image sources store row 0 at the top, but WebGL texture v=0 is the bottom
    // texel; without this the rendered result comes out vertically flipped.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.originalWidth = width;
    this.originalHeight = height;
    return this.originalTexture;
  }

  getOriginalTexture(): WebGLTexture | null {
    return this.originalTexture;
  }

  getOriginalSize(): { width: number; height: number } {
    return { width: this.originalWidth, height: this.originalHeight };
  }

  /**
   * Uploads a CPU-produced auxiliary texture (currently the face mask) under a
   * name. Uses the same UNPACK_FLIP_Y_WEBGL as setOriginalImage — without it the
   * mask would be vertically mirrored relative to the image it describes.
   */
  setAuxTexture(name: string, source: TexImageSource): WebGLTexture {
    const gl = this.gl;
    let tex = this.auxTextures.get(name) ?? null;
    if (!tex) {
      tex = gl.createTexture();
      if (!tex) throw new Error("補助テクスチャを作成できませんでした。");
      this.auxTextures.set(name, tex);
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  getAuxTexture(name: string): WebGLTexture | null {
    return this.auxTextures.get(name) ?? null;
  }

  clearAuxTexture(name: string): void {
    const tex = this.auxTextures.get(name);
    if (!tex) return;
    this.gl.deleteTexture(tex);
    this.auxTextures.delete(name);
  }

  /**
   * A 1x1 texture bound wherever a sampler must be populated but no real data
   * exists — sampling an unbound sampler2D is undefined behaviour, so this is
   * never optional.
   *
   * It is BLACK, not white, and that direction is deliberate: a shader that
   * forgets to gate on uHasFaceMask then degrades to "no face anywhere" (the
   * filter simply does nothing region-specific) instead of "every pixel is
   * skin", which would smooth or flatten the entire frame.
   */
  getPlaceholderMaskTexture(): WebGLTexture {
    const gl = this.gl;
    if (this.placeholderMaskTexture) return this.placeholderMaskTexture;
    const tex = gl.createTexture();
    if (!tex) throw new Error("テクスチャを作成できませんでした。");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.placeholderMaskTexture = tex;
    return tex;
  }

  /** Gets (creating or resizing as needed) the pooled render target for `key`. */
  acquireTarget(key: string, width: number, height: number): PooledTarget {
    const gl = this.gl;
    let entry = this.pool.get(key);
    if (!entry) {
      const texture = gl.createTexture();
      const framebuffer = gl.createFramebuffer();
      if (!texture || !framebuffer) throw new Error("フレームバッファを作成できませんでした。");
      entry = { texture, framebuffer, width: 0, height: 0, lastUsedFrame: this.frame };
      this.pool.set(key, entry);
    }

    if (entry.width !== width || entry.height !== height) {
      gl.bindTexture(gl.TEXTURE_2D, entry.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      gl.bindFramebuffer(gl.FRAMEBUFFER, entry.framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, entry.texture, 0);

      entry.width = width;
      entry.height = height;
    }

    entry.lastUsedFrame = this.frame;
    return entry;
  }

  /** Disposes pooled targets not used within the last `maxAgeFrames` frames. */
  evictStale(maxAgeFrames = 120): void {
    for (const [key, entry] of this.pool) {
      if (this.frame - entry.lastUsedFrame > maxAgeFrames) {
        this.disposeTarget(entry);
        this.pool.delete(key);
      }
    }
  }

  /** Disposes every pooled target whose key does not start with any of `keepPrefixes`. */
  evictExcept(keepPrefixes: string[]): void {
    for (const [key, entry] of this.pool) {
      if (!keepPrefixes.some((prefix) => key.startsWith(prefix))) {
        this.disposeTarget(entry);
        this.pool.delete(key);
      }
    }
  }

  private disposeTarget(entry: PooledTarget): void {
    this.gl.deleteTexture(entry.texture);
    this.gl.deleteFramebuffer(entry.framebuffer);
  }

  disposeAll(): void {
    for (const entry of this.pool.values()) {
      this.disposeTarget(entry);
    }
    this.pool.clear();
    if (this.originalTexture) {
      this.gl.deleteTexture(this.originalTexture);
      this.originalTexture = null;
    }
    for (const tex of this.auxTextures.values()) {
      this.gl.deleteTexture(tex);
    }
    this.auxTextures.clear();
    if (this.placeholderMaskTexture) {
      this.gl.deleteTexture(this.placeholderMaskTexture);
      this.placeholderMaskTexture = null;
    }
  }
}
