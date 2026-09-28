// Data layer. Talks to the REST API (backend/api) when it is up; otherwise falls back to the
// precomputed files (/products, /static, /reports served by vite.config.ts) — "offline demo mode".
// Every number shown in the UI comes from one of these two sources.
import type { Alert, CaseSummary, DistrictRow, FeatureCollection, Meta } from "./types";

export const API: string = (import.meta.env.VITE_API as string | undefined) ?? "http://127.0.0.1:8000";
export type Mode = "api" | "offline";
let mode: Mode | null = null;
const listeners = new Set<(m: Mode) => void>();

export function onMode(fn: (m: Mode) => void) { listeners.add(fn); if (mode) fn(mode); return () => listeners.delete(fn); }

export async function detectMode(timeoutMs = 1500): Promise<Mode> {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    const r = await fetch(`${API}/health`, { signal: ctl.signal });
    clearTimeout(t);
    mode = r.ok ? "api" : "offline";
  } catch {
    mode = "offline";
  }
  listeners.forEach((fn) => fn(mode!));
  return mode;
}

export function currentMode(): Mode { return mode ?? "offline"; }

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  // tolerate bare NaN from older result files (Python json); the writers now emit null
  return JSON.parse((await r.text()).replace(/(?<=[:\[,]\s*)(?:NaN|-?Infinity)(?=\s*[,\]}])/g, "null")) as T;
}
const base = () => (mode === "api" ? API : "");
const pad3 = (h: number) => String(h).padStart(3, "0");

export const getCases = () => (mode === "api" ? json<CaseSummary[]>(`${API}/cases`) : json<CaseSummary[]>("/products/index.json"));
export const getMeta = (c: string) => (mode === "api" ? json<Meta>(`${API}/cases/${c}`) : json<Meta>(`/products/${c}/meta.json`));
export const getReport = <T = any>(name: string) => json<T>(`${base()}/reports/${name}`);
export const staticUrl = (p: string) => `${base()}/static/${p}`;

export function fieldUrl(c: string, v: string, res: "12km" | "5km", lead: number): string {
  if (v === "strike") return `${base()}/products/${c}/strike_prob/${pad3(lead)}.png`;
  return `${base()}/products/${c}/fields_${res}/${v}/${pad3(lead)}.png`;
}

/** Tracks cut at `lead` (same rule as GET /cases/{c}/tracks?lead=). */
export function cutTracks(fc: FeatureCollection, lead: number): FeatureCollection {
  const out: FeatureCollection = { type: "FeatureCollection", features: [] };
  for (const f of fc.features) {
    const p = f.properties;
    if (p.kind === "member_track" || p.kind === "consensus") {
      const keep = (p.leads_h as number[]).map((h, i) => [h, i]).filter(([h]) => h <= lead).map(([, i]) => i);
      if (!keep.length) continue;
      const n = p.leads_h.length;
      const props: Record<string, any> = {};
      for (const [k, v] of Object.entries(p)) props[k] = Array.isArray(v) && v.length === n ? keep.map((i) => v[i]) : v;
      props.active = p.leads_h[keep[keep.length - 1]] === lead;
      out.features.push({ type: "Feature", properties: props,
        geometry: { type: "LineString", coordinates: keep.map((i) => f.geometry.coordinates[i]) } });
    } else {
      out.features.push({ ...f, properties: { ...p, active: p.lead_start_h <= lead && lead <= p.lead_end_h } });
    }
  }
  return out;
}

const trackCache = new Map<string, Promise<FeatureCollection>>();
export async function getTracks(c: string): Promise<FeatureCollection> {
  const k = `${mode}:${c}`;
  if (!trackCache.has(k)) {
    trackCache.set(k, mode === "api" ? json<FeatureCollection>(`${API}/cases/${c}/tracks`)
      : json<FeatureCollection>(`/products/${c}/tracks.geojson`));
  }
  return trackCache.get(k)!;
}

const alertCache = new Map<string, Promise<Alert[]>>();
function allAlerts(c: string): Promise<Alert[]> {
  const k = `${mode}:${c}`;
  if (!alertCache.has(k)) {
    alertCache.set(k, mode === "api"
      ? json<{ alerts: Alert[] }>(`${API}/alerts?case=${c}&limit=5000`).then((r) => r.alerts)
      : json<{ alerts: Alert[] }>(`/products/${c}/alerts.json`).then((r) => r.alerts));
  }
  return alertCache.get(k)!;
}
const RANK: Record<string, number> = { low: 1, moderate: 2, severe: 3 };
export const sortAlerts = (a: Alert[]) =>
  [...a].sort((x, y) => RANK[y.category] - RANK[x.category] || y.probability - x.probability || x.lead_h - y.lead_h);

export async function getAlerts(c: string, lead?: number): Promise<Alert[]> {
  const a = await allAlerts(c);
  return sortAlerts(lead === undefined ? a : a.filter((x) => x.lead_h === lead));
}

/** Alerts covering a point: API (exact 12 km alert region) or, offline, the region's bounding box. */
export async function alertsAt(c: string, lat: number, lon: number, lead?: number): Promise<{ alerts: Alert[]; exact: boolean }> {
  if (mode === "api") {
    const q = new URLSearchParams({ case: c, lat: String(lat), lon: String(lon) });
    if (lead !== undefined) q.set("lead", String(lead));
    const r = await json<{ alerts: Alert[] }>(`${API}/alerts?${q}`);
    return { alerts: r.alerts, exact: true };
  }
  const a = await getAlerts(c, lead);
  const hit = a.filter((x) => { const [x0, y0, x1, y1] = x.region_bbox; return lon >= x0 - 0.06 && lon <= x1 + 0.06 && lat >= y0 - 0.06 && lat <= y1 + 0.06; });
  return { alerts: hit, exact: false };
}

export async function getDistricts(c: string, lead?: number): Promise<DistrictRow[] | null> {
  if (mode !== "api") return null;                 // roll-up needs the API (alert_grid + district raster)
  const q = lead === undefined ? "" : `&lead=${lead}`;
  try { return (await json<{ districts: DistrictRow[] }>(`${API}/alerts/districts?case=${c}${q}`)).districts; }
  catch { return null; }
}
