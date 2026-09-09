export class ShaderCompileError extends Error {
  constructor(
    message: string,
    public readonly stage: "vertex" | "fragment" | "link",
  ) {
    super(message);
  }
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new ShaderCompileError("シェーダーオブジェクトを作成できませんでした。", "vertex");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "unknown error";
    gl.deleteShader(shader);
    throw new ShaderCompileError(log, type === gl.VERTEX_SHADER ? "vertex" : "fragment");
  }
  return shader;
}

/** A compiled+linked WebGL2 program with cached uniform locations. */
export class ShaderProgram {
  readonly program: WebGLProgram;
  private readonly gl: WebGL2RenderingContext;
  private readonly uniformLocations = new Map<string, WebGLUniformLocation | null>();

  constructor(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string) {
    this.gl = gl;
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    if (!program) throw new ShaderCompileError("プログラムを作成できませんでした。", "link");
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? "unknown link error";
      gl.deleteProgram(program);
      throw new ShaderCompileError(log, "link");
    }
    this.program = program;
  }

  use(): void {
    this.gl.useProgram(this.program);
  }

  private location(name: string): WebGLUniformLocation | null {
    if (this.uniformLocations.has(name)) {
      return this.uniformLocations.get(name) ?? null;
    }
    const loc = this.gl.getUniformLocation(this.program, name);
    this.uniformLocations.set(name, loc);
    return loc;
  }

  setFloat(name: string, value: number): void {
    const loc = this.location(name);
    if (loc) this.gl.uniform1f(loc, value);
  }

  setVec2(name: string, x: number, y: number): void {
    const loc = this.location(name);
    if (loc) this.gl.uniform2f(loc, x, y);
  }

  setVec4(name: string, x: number, y: number, z: number, w: number): void {
    const loc = this.location(name);
    if (loc) this.gl.uniform4f(loc, x, y, z, w);
  }

  setInt(name: string, value: number): void {
    const loc = this.location(name);
    if (loc) this.gl.uniform1i(loc, value);
  }

  dispose(): void {
    this.gl.deleteProgram(this.program);
    this.uniformLocations.clear();
  }
}
