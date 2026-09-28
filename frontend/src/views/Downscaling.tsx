import { useEffect, useRef, useState } from "react";
import { BarChart, Legend, LineChart, OKABE, Skeleton } from "../components/ui";
import { fieldUrl, getReport } from "../api";
import type { Meta } from "../types";

const MODELS = ["bicubic+lapse", "unet", "diffusion_mean", "diffusion_sample"];
const VARS = ["tp", "wind", "t2m", "msl"];
const img = (src: string) => new Promise<HTMLImageElement | null>((r) => { const i = new Image(); i.crossOrigin = "anonymous"; i.onload = () => r(i); i.onerror = () => r(null); i.src = src; });

export default function Downscaling({ meta, lead }: { meta: Meta; lead: number }) {
  const [v, setV] = useState("tp");
  const [swipe, setSwipe] = useState(50);
  const [ds, setDs] = useState<any>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const l5 = [...meta.leads_5km_h].reverse().find((h) => h <= lead) ?? 0;
  useEffect(() => { getReport("downscaling_results.json").then(setDs).catch(() => setDs(false)); }, []);

  useEffect(() => {
    let live = true;
    Promise.all([img(fieldUrl(meta.case, v, "12km", l5)), img(fieldUrl(meta.case, v, "5km", l5))]).then(([a, b]) => {
      const c = cv.current;
      if (!live || !c) return;
      const ctx = c.getContext("2d")!;
      ctx.clearRect(0, 0, c.width, c.height);
      const box = meta.bbox4d[0] ?? { lon_min: 80, lon_max: 92, lat_min: 10, lat_max: 24 };
      const [W, S, E, N] = meta.bounds;
      const cx = (box.lon_min + box.lon_max) / 2, cy = (box.lat_min + box.lat_max) / 2;
      const h = Math.max(box.lon_max - box.lon_min, box.lat_max - box.lat_min) / 2 + 1.5;
      const crop = (im: HTMLImageElement) => [((cx - h - W) / (E - W)) * im.width, ((N - (cy + h)) / (N - S)) * im.height,
        ((2 * h) / (E - W)) * im.width, ((2 * h) / (N - S)) * im.height] as const;
      ctx.imageSmoothingEnabled = false;
      if (a) { const [sx, sy, sw, sh] = crop(a); ctx.drawImage(a, sx, sy, sw, sh, 0, 0, c.width, c.height); }
      if (b) {
        const x = (swipe / 100) * c.width;
        ctx.save(); ctx.beginPath(); ctx.rect(x, 0, c.width - x, c.height); ctx.clip();
        const [sx, sy, sw, sh] = crop(b); ctx.drawImage(b, sx, sy, sw, sh, 0, 0, c.width, c.height);
        ctx.restore();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke();
      }
    });
    return () => { live = false; };
  }, [meta, v, l5, swipe]);

  const spec = ds?.spectra;
  const unit = v === "tp" ? "mm/6h" : v === "wind" ? "m/s" : v === "t2m" ? "degC" : "hPa";
  return (
    <section aria-label="Downscaling view">
      <div className="toolbar">
        <label>Variable <select value={v} onChange={(e) => setV(e.target.value)}>{VARS.map((x) => <option key={x}>{x}</option>)}</select></label>
        <label className="grow">Swipe 12 km ◀▶ 5 km <input type="range" min={0} max={100} value={swipe} onChange={(e) => setSwipe(Number(e.target.value))} aria-label="Swipe between 12 km and 5 km" /></label>
        <span className="note">+{l5} h (5 km exported every 12 h)</span>
      </div>
      <div className="grid2">
        <div className="card">
          <h2>12 km input (left) | 5 km U-Net, exact conservation (right)</h2>
          <canvas ref={cv} width={640} height={640} className="swipe" aria-label={`${v} at 12 km and 5 km, same colour scale`} />
          <Legend legend={meta.legends[v]} title={v} />
          <p className="note">Same colour scale on both sides. Zoomed to the 4-D box + 1.5°. Member: {meta.fields_member}.</p>
        </div>
        <div className="card">
          <h2>Peaks preserved? Power spectrum (SYNTHETIC TEST)</h2>
          {ds === null ? <Skeleton h={240} /> : !spec ? <p className="note">reports/downscaling_results.json not available.</p> :
            <LineChart logx logy xrev xlabel="wavelength (km)" ylabel={`power (${unit})²`}
              series={[{ name: "truth 5 km", color: "currentColor", x: spec.unet[v].wavelength_km, y: spec.unet[v].truth, dash: "5 3" },
                ...MODELS.map((m, i) => ({ name: m, color: OKABE[i], x: spec[m][v].wavelength_km, y: spec[m][v].pred }))]} />}
          {ds && <BarChart ylabel="p99 ratio (pred / truth)" ref1={1} groups={VARS}
            series={MODELS.map((m, i) => ({ name: m, color: OKABE[i], values: VARS.map((x) => ds.metrics[m][x]?.p99_ratio ?? null) }))} />}
          {ds && <BarChart ylabel="power ratio at 10 km (rain, wind)" ref1={1} groups={["tp", "wind"]}
            series={MODELS.map((m, i) => ({ name: m, color: OKABE[i], values: ["tp", "wind"].map((x) => ds.metrics[m][x]?.psd_ratio_10km ?? null) }))} />}
          <p className="note">From reports/downscaling_results.json (SYNTHETIC TEST cases, exact 5 km truth). The p99 ratio does not separate the models;
            the 10 km power ratio does (reports/DOWNSCALE_RESULTS.md).</p>
        </div>
      </div>
    </section>
  );
}
