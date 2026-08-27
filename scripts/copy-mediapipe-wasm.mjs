// Copies MediaPipe's vision wasm runtime out of node_modules into public/ so the
// app can serve it from its own origin. Nothing is ever fetched from a CDN — that
// would both break the offline/PWA story and put a third-party origin in the
// request path of an app whose whole premise is that images never leave the device.
//
// Runs from postinstall and prebuild, because the .js glue and the .wasm binary are
// a matched pair: a dependency bump that updated one without the other would fail at
// runtime in a way that is very hard to read.
//
// Only the SIMD build is copied. Every browser this app targets (current Chrome /
// Edge, Firefox 89+, Safari 16.4+) has wasm SIMD; on anything older the fetch 404s
// and face detection degrades to the shader-only heuristic, which is a supported path.
import { copyFileSync, mkdirSync, existsSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dest = join(root, "public", "mediapipe", "wasm");

const FILES = ["vision_wasm_internal.js", "vision_wasm_internal.wasm"];

if (!existsSync(src)) {
  console.warn("[copy-mediapipe-wasm] @mediapipe/tasks-vision not installed; skipping.");
  process.exit(0);
}

mkdirSync(dest, { recursive: true });
let total = 0;
for (const file of FILES) {
  const from = join(src, file);
  if (!existsSync(from)) {
    console.error(`[copy-mediapipe-wasm] missing ${file} — face detection will fall back.`);
    continue;
  }
  copyFileSync(from, join(dest, file));
  total += statSync(from).size;
}
console.log(`[copy-mediapipe-wasm] copied ${FILES.length} files (${(total / 1024 / 1024).toFixed(1)} MB) to public/mediapipe/wasm`);
