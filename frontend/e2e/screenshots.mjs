// Screenshots of the 4 dashboard views for docs/PITCH.md:  node e2e/screenshots.mjs
// (API and dev server must be running).
import { chromium } from "@playwright/test";
const out = new URL("../../docs/figures/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const b = await chromium.launch({ channel: "msedge", args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
const shots = [
  ["ui_operations", "/?case=amphan_replay&view=ops&lead=96"],
  ["ui_downscaling", "/?case=amphan_replay&view=ds&lead=96"],
  ["ui_alerts", "/?case=amphan_replay&view=alerts&lead=96"],
  ["ui_performance", "/?case=amphan_replay&view=perf"],
];
for (const [name, url] of shots) {
  await p.goto("http://127.0.0.1:5173" + url);
  await p.waitForLoadState("networkidle");
  await p.waitForTimeout(3500);
  if (name === "ui_alerts") { await p.locator("table").first().locator("tbody tr").first().click(); await p.waitForTimeout(800); }
  await p.screenshot({ path: out + name + ".png", fullPage: true });
  console.log("wrote", name);
}
await p.goto("http://127.0.0.1:5173/?case=amphan_replay&view=ops&lead=96");
await p.waitForTimeout(3000);
await p.getByText("3-D (time-extruded)").click();
await p.waitForTimeout(3000);
await p.screenshot({ path: out + "ui_operations_3d.png" });
await p.setViewportSize({ width: 390, height: 844 });
await p.waitForTimeout(2000);
await p.screenshot({ path: out + "ui_mobile.png", fullPage: true });
console.log("wrote 3d + mobile");
await b.close();
