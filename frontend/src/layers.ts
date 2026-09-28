import { BitmapLayer, GeoJsonLayer, PathLayer, PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { Alert, Feature, FeatureCollection, Meta } from "./types";
import { hex2rgb, OKABE } from "./components/ui";

const NEAREST = { minFilter: "nearest", magFilter: "nearest" } as const;
const HOUR_M = 2500;                    // 4-D box: 1 h of lead = 2.5 km of extrusion height

export function field(id: string, url: string, meta: Meta, opacity = 0.85) {
  const [w, s, e, n] = meta.bounds;
  return new BitmapLayer({ id, image: url, bounds: [w, s, e, n], opacity, textureParameters: NEAREST as any });
}

export function outline(id: string, data: any, color: [number, number, number, number] = [255, 255, 255, 190]) {
  return new GeoJsonLayer({ id, data, stroked: true, filled: false, getLineColor: color, lineWidthUnits: "pixels", getLineWidth: 1.2 });
}

export function memberTracks(fc: FeatureCollection) {
  const data = fc.features.filter((f) => f.properties.kind === "member_track");
  return new PathLayer<Feature>({
    id: "members", data, widthUnits: "pixels", pickable: true,
    getPath: (f) => f.geometry.coordinates, getWidth: (f) => (f.properties.main ? 2 : 1),
    getColor: (f) => (f.properties.main ? [230, 237, 243, 200] : [180, 190, 200, 90]),
  });
}

export function consensus(fc: FeatureCollection) {
  const c = fc.features.find((f) => f.properties.kind === "consensus");
  if (!c) return [];
  const pts = c.geometry.coordinates.map((xy: number[], i: number) => ({ xy, r: c.properties.spread_km[i] * 1000, n: c.properties.member_count[i], lead: c.properties.leads_h[i] }));
  return [
    new ScatterplotLayer<any>({ id: "cone", data: pts, getPosition: (d) => d.xy, getRadius: (d) => Math.max(d.r, 5000),
      getFillColor: [...hex2rgb(OKABE[1]), 40], stroked: true, getLineColor: [...hex2rgb(OKABE[1]), 110],
      lineWidthUnits: "pixels", getLineWidth: 1, pickable: true }),
    new PathLayer<any>({ id: "consensus", data: [c], getPath: (f) => f.geometry.coordinates, getColor: [...hex2rgb(OKABE[1]), 255],
      widthUnits: "pixels", getWidth: 3 }),
  ];
}

export function box4d(fc: FeatureCollection, extruded: boolean) {
  const data = fc.features.filter((f) => f.properties.kind === "bbox4d");
  return new PolygonLayer<Feature>({
    id: "bbox4d", data, pickable: true, extruded, wireframe: true, filled: extruded, stroked: true,
    // base at the box's first lead, top at its last lead: time is the vertical axis
    getPolygon: (f) => f.geometry.coordinates[0].map(([x, y]: number[]) => [x, y, extruded ? f.properties.lead_start_h * HOUR_M : 0]),
    getElevation: (f) => (f.properties.lead_end_h - f.properties.lead_start_h) * HOUR_M,
    getFillColor: (f) => [...hex2rgb(OKABE[3]), f.properties.active ? 70 : 25],
    getLineColor: (f) => [...hex2rgb(OKABE[3]), f.properties.active ? 255 : 120],
    lineWidthUnits: "pixels", getLineWidth: 2,
  });
}

export function alertPins(alerts: Alert[], selected?: string, onPick?: (a: Alert) => void) {
  return [
    new ScatterplotLayer<Alert>({ id: "alert-radius", data: alerts, getPosition: (a) => [a.pinpoint.lon, a.pinpoint.lat],
      getRadius: (a) => a.impact_radius_km * 1000, radiusUnits: "meters", getFillColor: (a) => [...hex2rgb(a.colour), 90],
      stroked: true, getLineColor: (a) => [...hex2rgb(a.colour), 255], lineWidthUnits: "pixels", getLineWidth: 1 }),
    new ScatterplotLayer<Alert>({ id: "alert-pins", data: alerts, pickable: true, onClick: (info: any) => { if (info.object && onPick) onPick(info.object); return true; }, getPosition: (a) => [a.pinpoint.lon, a.pinpoint.lat],
      radiusUnits: "pixels", getRadius: (a) => (a.id === selected ? 9 : 6), getFillColor: (a) => hex2rgb(a.colour),
      stroked: true, getLineColor: [0, 0, 0, 255], lineWidthUnits: "pixels", getLineWidth: (a) => (a.id === selected ? 3 : 1),
      updateTriggers: { getRadius: selected, getLineWidth: selected } }),
  ];
}

export function tooltip(info: any) {
  const o = info.object;
  if (!o) return null;
  if (o.reason) return { text: `${o.category.toUpperCase()} ${o.kind}\n${o.reason}\n+${o.lead_h} h, ${o.valid_time}` };
  if (o.properties?.kind === "member_track") return { text: `member ${o.properties.member}${o.properties.main ? " (main track)" : ""}` };
  if (o.properties?.kind === "bbox4d") {
    const p = o.properties;
    return { text: `4-D box: +${p.lead_start_h}–${p.lead_end_h} h\n${p.level_hPa_bottom}–${p.level_hPa_top} hPa\n≥ ${p.member_count_min}/${p.n_members} members` };
  }
  if (o.r !== undefined) return { text: `consensus +${o.lead} h\nspread ${(o.r / 1000).toFixed(0)} km, ${o.n} members` };
  return null;
}

/** IBTrACS best track (REAL events): white path + fixes; the fix at the current valid time is ringed. */
export function ibtracs(meta: Meta, lead: number) {
  const fixes = meta.ibtracs ?? [];
  const vt = new Date((meta.valid_times[meta.leads_h.indexOf(lead)] ?? meta.valid_times[0]).replace(" ", "T") + "Z").getTime();
  const path = fixes.map((f) => [f.lon, f.lat]);
  const now = fixes.filter((f) => Math.abs(new Date(f.time.replace(" ", "T") + "Z").getTime() - vt) < 3 * 3600e3);
  return [
    new PathLayer<any>({ id: "ibtracs", data: [{ path }], getPath: (d) => d.path, getColor: [255, 255, 255, 230], widthUnits: "pixels",
      getWidth: 2.5 }),
    new ScatterplotLayer<any>({ id: "ibtracs-fixes", data: fixes, getPosition: (f) => [f.lon, f.lat], radiusUnits: "pixels", getRadius: 2.5,
      getFillColor: [255, 255, 255, 200], pickable: true }),
    new ScatterplotLayer<any>({ id: "ibtracs-now", data: now, getPosition: (f) => [f.lon, f.lat], radiusUnits: "pixels", getRadius: 7,
      stroked: true, filled: false, getLineColor: [255, 255, 255, 255], lineWidthUnits: "pixels", getLineWidth: 2 }),
  ];
}
