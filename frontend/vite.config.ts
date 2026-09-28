/// <reference types="vitest" />
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// Serves the precomputed pipeline outputs next to the app, so the dashboard also works with the API
// down ("offline demo mode"): /products -> ../backend/products, /static -> ../backend/static,
// /reports -> ../reports (*.json only).
const ROOT = path.resolve(__dirname, "..");
const MOUNTS: Record<string, { dir: string; allow?: RegExp }> = {
  "/products/": { dir: path.join(ROOT, "backend/products") },
  "/static/": { dir: path.join(ROOT, "backend/static") },
  "/reports/": { dir: path.join(ROOT, "reports"), allow: /^[\w.-]+\.json$/ },
};
const TYPES: Record<string, string> = { ".json": "application/json", ".geojson": "application/geo+json", ".png": "image/png" };

function products(): Plugin {
  const handler = (req: any, res: any, next: any) => {
    const url = decodeURIComponent((req.url || "").split("?")[0]);
    for (const [prefix, m] of Object.entries(MOUNTS)) {
      if (!url.startsWith(prefix)) continue;
      const rel = url.slice(prefix.length);
      const file = path.resolve(m.dir, rel);
      if (!file.startsWith(m.dir) || (m.allow && !m.allow.test(rel)) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.statusCode = 404;
        return res.end("not found");
      }
      res.setHeader("Content-Type", TYPES[path.extname(file)] || "application/octet-stream");
      return fs.createReadStream(file).pipe(res);
    }
    next();
  };
  return {
    name: "sih-products",
    configureServer: (s) => { s.middlewares.use(handler); },
    configurePreviewServer: (s) => { s.middlewares.use(handler); },
  };
}

export default defineConfig({
  plugins: [react(), products()],
  server: { port: 5173, host: "127.0.0.1" },
  preview: { port: 4173, host: "127.0.0.1" },
  build: { chunkSizeWarningLimit: 2000 },
  test: { environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"] },
});
