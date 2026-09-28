// Smoke test (BRIEF3 Phase D): load the page, move the lead slider, see an alert.
// Runs in jsdom with the API down (offline demo mode); fetch() reads the real exported product
// files from backend/products and backend/static. The WebGL map is replaced by a stub.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import fs from "node:fs";
import path from "node:path";
import { vi } from "vitest";
import App from "../App";
import { cutTracks } from "../api";

vi.mock("../components/MapView", () => ({ default: (p: any) => <div data-testid="map" aria-label={p.label} data-layers={p.layers.length} /> }));

const ROOT = path.resolve(__dirname, "../../..");
const MOUNTS: Record<string, string> = { "/products/": "backend/products", "/static/": "backend/static", "/reports/": "reports" };

beforeAll(() => {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.startsWith("http")) throw new Error("API down");          // -> offline mode
    for (const [pre, dir] of Object.entries(MOUNTS)) {
      if (url.startsWith(pre)) {
        const f = path.join(ROOT, dir, url.slice(pre.length).split("?")[0]);
        if (fs.existsSync(f)) return new Response(fs.readFileSync(f, "utf8"), { status: 200 });
      }
    }
    return new Response("not found", { status: 404 });
  });
  window.history.replaceState({}, "", "/?case=amphan_replay&view=alerts");
  HTMLCanvasElement.prototype.getContext = (() => null) as any;
});

test("load, move the slider, see an alert", async () => {
  render(<App />);
  expect(await screen.findByText("○ offline demo")).toBeTruthy();
  expect(await screen.findByText("SYNTHETIC", { selector: ".badge" })).toBeTruthy();
  const slider = await screen.findByLabelText("Lead time (hours)", {}, { timeout: 8000 });
  fireEvent.change(slider, { target: { value: "96" } });
  await waitFor(() => expect(screen.getByText(/Alerts \(\+96 h\)/)).toBeTruthy());
  const table = screen.getAllByRole("table")[0];
  await waitFor(() => expect(within(table).getAllByText(/^(severe|moderate|low)$/).length).toBeGreaterThan(0));
  fireEvent.click(within(table).getAllByRole("row")[1]);
  await waitFor(() => expect(screen.getAllByText(/^P\(.+\) = \d\.\d\d within 50 km$/).length).toBeGreaterThan(0));
});

test("operations view renders the map with layers and legends", async () => {
  window.history.replaceState({}, "", "/?case=cyc_04&view=ops&lead=48");
  localStorage.clear();
  render(<App />);
  const map = await screen.findByTestId("map", {}, { timeout: 8000 });
  expect(Number(map.dataset.layers)).toBeGreaterThan(3);
  expect(screen.getByLabelText("Strike probability legend")).toBeTruthy();
});

test("cutTracks keeps only leads <= the slider lead", () => {
  const fc = { type: "FeatureCollection" as const, features: [
    { type: "Feature" as const, geometry: { type: "LineString", coordinates: [[80, 10], [81, 11], [82, 12]] },
      properties: { kind: "member_track", leads_h: [0, 6, 12] } },
    { type: "Feature" as const, geometry: { type: "Polygon", coordinates: [] }, properties: { kind: "bbox4d", lead_start_h: 6, lead_end_h: 12 } }] };
  const c = cutTracks(fc, 6);
  expect(c.features[0].geometry.coordinates).toHaveLength(2);
  expect(c.features[0].properties.active).toBe(true);
  expect(c.features[1].properties.active).toBe(true);
  expect(cutTracks(fc, 0).features[1].properties.active).toBe(false);
});
