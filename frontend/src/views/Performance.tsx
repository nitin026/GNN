import { useEffect, useState } from "react";
import { BarChart, LineChart, OKABE, Skeleton } from "../components/ui";
import { getReport } from "../api";

const METHODS: [string, string][] = [["v2", "tracker v2"], ["te_style", "TE-style"], ["gnn_full", "GNN full"], ["gnn_temporal", "GNN temporal"], ["gnn_none", "node-only MLP"]];

function mean(xs: (number | null | undefined)[]) {
  const v = xs.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

export default function Performance() {
  const [g, setG] = useState<any>(null);
  const [t, setT] = useState<any>(null);
  const [split, setSplit] = useState<"test" | "val">("test");
  useEffect(() => { getReport("gnn_results.json").then(setG).catch(() => setG(false)); getReport("demo_timing.json").then(setT).catch(() => setT(false)); }, []);
  if (g === null) return <Skeleton h={400} label="Loading metrics" />;
  if (!g) return <p className="note">reports/gnn_results.json not available.</p>;

  const cases = Object.entries(g.cases as Record<string, any>).filter(([, c]) => c.split === split);
  const m = (meth: string, k: string) => mean(cases.map(([, c]) => c[meth]?.[k]));
  const best = g.amphan_real.gnn_temporal;
  const timing = t ? Object.values(t as Record<string, any>).find((r: any) => r.graph === "built") as any : null;
  const bands = Object.keys(g.reliability_truth);
  return (
    <section aria-label="Model performance view">
      <div className="toolbar">
        <label>Split <select value={split} onChange={(e) => setSplit(e.target.value as any)}><option value="test">TEST (never tuned on)</option><option value="val">VAL</option></select></label>
        <span className="badge syn">SYNTHETIC</span><span className="note">{cases.map(([k]) => k).join(", ")}</span>
      </div>
      <div className="cards">
        <div className="card metric"><div className="k">CSI · GNN full</div><div className="v">{m("gnn_full", "csi")?.toFixed(2)}</div><div className="note">v2 {m("v2", "csi")?.toFixed(2)} · TE-style {m("te_style", "csi")?.toFixed(2)}</div></div>
        <div className="card metric"><div className="k">FAR · GNN full</div><div className="v">{m("gnn_full", "far")?.toFixed(2)}</div><div className="note">v2 {m("v2", "far")?.toFixed(2)} · TE-style {m("te_style", "far")?.toFixed(2)}</div></div>
        <div className="card metric"><div className="k">Spurious tracks / case</div><div className="v">{m("gnn_full", "spurious_tracks")?.toFixed(1)}</div><div className="note">v2 {m("v2", "spurious_tracks")?.toFixed(1)} · TE-style {m("te_style", "spurious_tracks")?.toFixed(0)}</div></div>
        <div className="card metric"><div className="k">IoU · GNN full</div><div className="v">{m("gnn_full", "iou")?.toFixed(2)}</div><div className="note">v2 {m("v2", "iou")?.toFixed(2)} (object-level, not re-segmented)</div></div>
        <div className="card metric real"><div className="k">Amphan track error vs IBTrACS <span className="badge real">REAL</span></div><div className="v">{best.err_mean_km.toFixed(0)} km</div>
          <div className="note">median {best.err_median_km.toFixed(0)} km, {best.n_matched} fixes (GNN temporal) · v2 {g.amphan_real.v2.err_mean_km.toFixed(0)} · TE-style {g.amphan_real.te_style.err_mean_km.toFixed(0)} km</div></div>
        {timing && <div className="card metric"><div className="k">One forecast cycle (CPU)</div><div className="v">{timing.total_seconds.toFixed(0)} s</div>
          <div className="note">{timing.members} members × {timing.leads} leads · peak RAM {(timing.peak_rss_mb / 1000).toFixed(1)} GB</div></div>}
      </div>
      <div className="grid2">
        <div className="card"><h2>Detection skill ({split.toUpperCase()} mean)</h2>
          <BarChart ylabel="score" groups={["CSI", "POD", "FAR", "IoU"]}
            series={METHODS.map(([k, n], i) => ({ name: n, color: OKABE[i], values: ["csi", "pod", "far", "iou"].map((x) => m(k, x)) }))} /></div>
        <div className="card"><h2>Amphan 2020: ERA5 vs IBTrACS (REAL)</h2>
          <BarChart ylabel="track error (km)" groups={["mean", "median", "max"]}
            series={METHODS.filter(([k]) => g.amphan_real[k]).map(([k, n], i) => ({ name: n, color: OKABE[i], values: ["err_mean_km", "err_median_km", "err_max_km"].map((x) => g.amphan_real[k][x]) }))} /></div>
        <div className="card"><h2>Reliability of GNN event probability vs truth (TEST)</h2>
          <LineChart xlabel="forecast probability" ylabel="observed frequency"
            series={[{ name: "perfect", color: "currentColor", x: [0, 1], y: [0, 1], dash: "4 3" },
              ...bands.map((b, i) => ({ name: b, color: OKABE[i + 4], x: g.reliability_truth[b].map((r: number[]) => r[0]), y: g.reliability_truth[b].map((r: number[]) => r[1]) }))]} />
          <p className="note">Probabilities are nearly binary and over-confident at long leads (p≈1 verifies {Math.round(100 * g.reliability_truth[bands[bands.length - 1]].slice(-1)[0][1])} % at {bands[bands.length - 1]}). Calibration is BRIEF4 Phase 3.</p></div>
        <div className="card"><h2>Per case ({split.toUpperCase()})</h2>
          <div className="tablewrap"><table><thead><tr><th scope="col">case</th><th scope="col">method</th><th scope="col">CSI</th><th scope="col">FAR</th><th scope="col">IoU</th><th scope="col">spur.</th></tr></thead>
            <tbody>{cases.flatMap(([cid, c]) => METHODS.slice(0, 3).map(([k, n]) => <tr key={cid + k}><td>{cid}</td><td>{n}</td><td>{c[k].csi.toFixed(2)}</td><td>{c[k].far.toFixed(2)}</td><td>{c[k].iou.toFixed(2)}</td><td>{c[k].spurious_tracks}</td></tr>))}</tbody></table></div></div>
      </div>
      <p className="note">All numbers read from reports/gnn_results.json and reports/demo_timing.json. Synthetic cases have exact labels; the Amphan row is real ERA5 reanalysis.</p>
    </section>
  );
}
