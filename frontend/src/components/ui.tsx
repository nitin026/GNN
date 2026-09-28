import { useEffect, useState } from "react";
import type { Legend as LegendT } from "../types";

// Okabe–Ito colour-blind-safe palette for categorical series (methods, models, tracks).
export const OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"];
export const SEV: Record<string, string> = { low: "#FFD400", moderate: "#FF8C00", severe: "#D7191C" }; // IMD codes
export const hex2rgb = (h: string): [number, number, number] =>
  [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

export function Badge({ synthetic, text }: { synthetic: boolean; text: string }) {
  return <span className={`badge ${synthetic ? "syn" : "real"}`} title={synthetic ? "Synthetic event with exact ground truth — not a real observation or forecast" : "Real data"}>{text}</span>;
}

export function Legend({ legend, title }: { legend?: LegendT; title: string }) {
  if (!legend) return null;
  return (
    <div className="legend" aria-label={`${title} legend`}>
      <span>{title}</span>
      <span>{legend.vmin}</span>
      <span className="bar" style={{ background: `linear-gradient(90deg, ${legend.stops.map((s) => s.color).join(",")})` }} />
      <span>{legend.vmax} {legend.units}</span>
    </div>
  );
}

export function Skeleton({ h = 200, label = "Loading" }: { h?: number; label?: string }) {
  return <div className="skeleton" style={{ height: h }} role="status" aria-label={label}><span className="sr">{label}…</span></div>;
}

export function LeadSlider({ lead, leads, onChange, validTime }: { lead: number; leads: number[]; onChange: (l: number) => void; validTime?: string }) {
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      onChange(leads[(leads.indexOf(lead) + 1) % leads.length]);
    }, 800);
    return () => clearInterval(id);
  }, [playing, lead, leads, onChange]);
  const max = leads[leads.length - 1] ?? 240;
  return (
    <div className="slider">
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"} aria-pressed={playing}>{playing ? "❚❚" : "▶"}</button>
      <input type="range" min={leads[0] ?? 0} max={max} step={6} value={lead} aria-label="Lead time (hours)"
        aria-valuetext={`plus ${lead} hours`} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="leadtxt" aria-live="polite">+{lead} h{validTime ? ` · valid ${validTime.slice(0, 16)} UTC` : ""}</span>
    </div>
  );
}

// ---------------------------------------------------------------- tiny SVG charts (no chart lib)
export interface Series { name: string; color: string; x: number[]; y: number[]; dash?: string }

export function LineChart({ series, xlabel, ylabel, logx = false, logy = false, h = 240, xrev = false }:
  { series: Series[]; xlabel: string; ylabel: string; logx?: boolean; logy?: boolean; h?: number; xrev?: boolean }) {
  const W = 520, H = h, L = 56, R = 12, T = 12, B = 40;
  const fx = (v: number) => (logx ? Math.log10(v) : v), fy = (v: number) => (logy ? Math.log10(Math.max(v, 1e-12)) : v);
  const xs = series.flatMap((s) => s.x.map(fx)), ys = series.flatMap((s) => s.y.map(fy)).filter(Number.isFinite);
  if (!xs.length || !ys.length) return null;
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  if (xrev) [x0, x1] = [x1, x0];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (v: number) => L + ((fx(v) - x0) / (x1 - x0 || 1)) * (W - L - R);
  const py = (v: number) => T + (1 - (fy(v) - y0) / (y1 - y0 || 1)) * (H - T - B);
  const ticks = (a: number, b: number, log: boolean) => {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    if (log) { const out = []; for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) out.push(10 ** e); return out; }
    return Array.from({ length: 5 }, (_, i) => lo + (i * (hi - lo)) / 4);
  };
  const fmt = (v: number) => (Math.abs(v) >= 1e4 ? v.toExponential(0) : Math.abs(v) >= 100 || v === 0 || Number.isInteger(v) ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2));
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${ylabel} vs ${xlabel}`}>
        {ticks(x0, x1, logx).map((t) => <g key={`x${t}`}><line x1={px(t)} x2={px(t)} y1={T} y2={H - B} className="grid" /><text x={px(t)} y={H - B + 14} textAnchor="middle">{fmt(t)}</text></g>)}
        {ticks(y0, y1, logy).map((t) => <g key={`y${t}`}><line x1={L} x2={W - R} y1={py(t)} y2={py(t)} className="grid" /><text x={L - 6} y={py(t) + 4} textAnchor="end">{fmt(t)}</text></g>)}
        {series.map((s) => <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash}
          points={s.x.map((x, i) => `${px(x)},${py(s.y[i])}`).filter((_, i) => Number.isFinite(fy(s.y[i]))).join(" ")} />)}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="axis">{xlabel}</text>
        <text x={14} y={(T + H - B) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(T + H - B) / 2})`} className="axis">{ylabel}</text>
      </svg>
      <figcaption className="keys">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</figcaption>
    </figure>
  );
}

export function BarChart({ groups, series, ylabel, ref1, h = 240 }:
  { groups: string[]; series: { name: string; color: string; values: (number | null)[] }[]; ylabel: string; ref1?: number; h?: number }) {
  const W = 520, H = h, L = 48, R = 8, T = 10, B = 44;
  const vals = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const ymax = Math.max(...vals, ref1 ?? 0) * 1.1 || 1;
  const gw = (W - L - R) / groups.length, bw = (gw * 0.8) / series.length;
  const py = (v: number) => T + (1 - v / ymax) * (H - T - B);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ylabel}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => <g key={f}><line x1={L} x2={W - R} y1={py(f * ymax)} y2={py(f * ymax)} className="grid" /><text x={L - 5} y={py(f * ymax) + 4} textAnchor="end">{(f * ymax).toFixed(2)}</text></g>)}
        {ref1 !== undefined && <line x1={L} x2={W - R} y1={py(ref1)} y2={py(ref1)} stroke="currentColor" strokeDasharray="4 3" />}
        {groups.map((g, gi) => (
          <g key={g}>
            {series.map((s, si) => s.values[gi] === null ? null : (
              <rect key={s.name} x={L + gi * gw + gw * 0.1 + si * bw} y={py(s.values[gi]!)} width={bw - 1} height={H - B - py(s.values[gi]!)} fill={s.color}>
                <title>{`${s.name} · ${g}: ${s.values[gi]!.toFixed(3)}`}</title>
              </rect>))}
            <text x={L + gi * gw + gw / 2} y={H - B + 14} textAnchor="middle">{g}</text>
          </g>))}
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(T + H - B) / 2})`} className="axis">{ylabel}</text>
      </svg>
      <figcaption className="keys">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</figcaption>
    </figure>
  );
}
