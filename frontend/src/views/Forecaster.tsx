import { useEffect, useRef, useState } from "react";
import { Legend, LeadSlider } from "../components/ui";
import { cutTracks, fieldUrl, getTracks, layerUrl, staticUrl } from "../api";
import type { FeatureCollection, Meta } from "../types";

// Forecaster view (BRIEF4 Phase 6): the same lead, side by side — raw ensemble, EFI, our tracker,
// and the truth (SYNTHETIC exact mask, or the IBTrACS best track for REAL events). Plain canvases
// (no WebGL) so four panels stay light.
const img = (src: string) => new Promise<HTMLImageElement | null>((r) => { const i = new Image(); i.onload = () => r(i); i.onerror = () => r(null); i.src = src; });
let outline: any = null;

type Overlay = (ctx: CanvasRenderingContext2D, pr: (lon: number, lat: number) => [number, number]) => void;

function Panel({ title, sub, src, meta, overlay, legend }: { title: string; sub: string; src: string | null; meta: Meta; overlay?: Overlay; legend?: JSX.Element | null }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let live = true;
    (async () => {
      if (!outline) outline = await (await fetch(staticUrl("india_outline.geojson"))).json().catch(() => null);
      const im = src ? await img(src) : null;
      const c = cv.current;
      if (!live || !c) return;
      const ctx = c.getContext("2d")!;
      const [w0, s0, e0, n0] = [62, 2, 98, 38];                 // view (deg)
      const [W, S, E, N] = meta.bounds;
      const pr = (lon: number, lat: number): [number, number] => [(lon - w0) / (e0 - w0) * c.width, (n0 - lat) / (n0 - s0) * c.height];
      ctx.fillStyle = "#0b1117";
      ctx.fillRect(0, 0, c.width, c.height);
      if (im) {
        ctx.imageSmoothingEnabled = false;
        const sx = (w0 - W) / (E - W) * im.width, sw = (e0 - w0) / (E - W) * im.width;
        const sy = (N - n0) / (N - S) * im.height, sh = (n0 - s0) / (N - S) * im.height;
        ctx.drawImage(im, sx, sy, sw, sh, 0, 0, c.width, c.height);
      }
      if (outline) {
        ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 1;
        for (const f of outline.features) for (const line of f.geometry.coordinates) {
          ctx.beginPath(); line.forEach(([x, y]: number[], i: number) => { const [a, b] = pr(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
        }
      }
      overlay?.(ctx, pr);
    })();
    return () => { live = false; };
  }, [src, meta, overlay]);
  return (
    <figure className="card fpanel">
      <h2>{title}</h2>
      <canvas ref={cv} width={420} height={420} aria-label={`${title}: ${sub}`} />
      <figcaption className="note">{sub}</figcaption>
      {legend}
    </figure>
  );
}

export default function Forecaster({ meta, lead, setLead }: { meta: Meta; lead: number; setLead: (l: number) => void }) {
  const [tracks, setTracks] = useState<FeatureCollection | null>(null);
  useEffect(() => { getTracks(meta.case).then(setTracks); }, [meta.case]);
  const cut = tracks ? cutTracks(tracks, lead) : null;
  const L_ = meta.layers ?? {};
  const ti = meta.leads_h.indexOf(lead);
  const vt = meta.valid_times[ti];
  const drawTracks: Overlay = (ctx, pr) => {
    for (const f of cut?.features ?? []) {
      const p = f.properties;
      if (p.kind !== "member_track" && p.kind !== "consensus") continue;
      ctx.strokeStyle = p.kind === "consensus" ? "#56B4E9" : "rgba(220,220,220,.45)";
      ctx.lineWidth = p.kind === "consensus" ? 2.5 : 0.8;
      ctx.beginPath(); f.geometry.coordinates.forEach(([x, y]: number[], i: number) => { const [a, b] = pr(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
    }
  };
  const drawTruth: Overlay = (ctx, pr) => {
    const fx = meta.ibtracs ?? [];
    if (!fx.length) return;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); fx.forEach((f, i) => { const [a, b] = pr(f.lon, f.lat); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
    const t = vt ? new Date(vt.replace(" ", "T") + "Z").getTime() : 0;
    const now = fx.find((f) => Math.abs(new Date(f.time.replace(" ", "T") + "Z").getTime() - t) < 3 * 3600e3);
    if (now) { const [a, b] = pr(now.lon, now.lat); ctx.beginPath(); ctx.arc(a, b, 6, 0, 7); ctx.stroke(); }
  };
  const anomVar = "anom";
  return (
    <section aria-label="Forecaster view">
      <div className="toolbar"><LeadSlider lead={lead} leads={meta.leads_h} onChange={setLead} validTime={vt} /></div>
      <div className="grid4">
        <Panel meta={meta} title="1 · Raw ensemble" src={fieldUrl(meta.case, anomVar, "12km", lead)}
          sub={`hazard z-anomaly of ${meta.fields_member}; grey = every member's tracked object path so far`} overlay={drawTracks}
          legend={<Legend legend={meta.legends.anom} title="z" />} />
        <Panel meta={meta} title="2 · EFI" src={L_.efi ? layerUrl(meta.case, "efi", lead) : null}
          sub={L_.efi ? "Extreme Forecast Index of the hazard variable (whole ensemble)" : "not available (single run: no ensemble)"}
          legend={L_.efi ? <Legend legend={meta.legends.efi} title="EFI" /> : null} />
        <Panel meta={meta} title="3 · Our tracker" src={L_.strike_cal ? layerUrl(meta.case, "strike_cal", lead) : fieldUrl(meta.case, "strike", "12km", lead)}
          sub={`${L_.strike_cal ? "calibrated" : "raw"} strike probability + consensus (blue); ${meta.tracker}`} overlay={drawTracks}
          legend={<Legend legend={meta.legends.strike} title="P" />} />
        <Panel meta={meta} title={meta.synthetic ? "4 · Truth (SYNTHETIC exact mask)" : "4 · Truth (IBTrACS best track)"}
          src={L_.truth ? layerUrl(meta.case, "truth", lead) : null}
          sub={meta.synthetic ? "exact event mask at this valid time (IMD daily labels for heat/cold)" : meta.ibtracs?.length ? "best track; circle = fix at this valid time" : "no independent truth for this case"}
          overlay={drawTruth} />
      </div>
    </section>
  );
}
