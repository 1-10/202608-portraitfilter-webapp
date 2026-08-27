import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  // Each test drives a real WebGL context AND lazily pulls the ~15MB MediaPipe
  // wasm + face model from the single preview server. Too many concurrent workers
  // saturate both and produce spurious load-timeout failures.
  workers: 2,
  reporter: "list",
  expect: { timeout: 15_000 },
  timeout: 90_000,
  use: {
    baseURL: "http://localhost:4173",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run preview -- --port 4173",
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        },
      },
    },
  ],
});
