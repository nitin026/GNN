import { defineConfig } from "@playwright/test";

// Uses the system Microsoft Edge (channel "msedge"), so no browser download is needed.
// Start the API (python -m backend.api) and the dev server (npm run dev) first, or let
// `webServer` start the dev server; without the API the app runs in offline demo mode.
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:5173", channel: "msedge", headless: true,
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] } },
  webServer: { command: "npx vite --port 5173", url: "http://127.0.0.1:5173", reuseExistingServer: true, timeout: 60_000 },
});
