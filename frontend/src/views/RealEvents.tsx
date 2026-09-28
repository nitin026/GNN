import { useEffect, useState } from "react";
import { getEvents } from "../api";
import { Skeleton } from "../components/ui";
import type { RealEvent } from "../types";

// Real-event pages (BRIEF4 Phase 6): the locked REAL split with the BRIEF4 Phase 1 scores; storms that
// have a dashboard case open it (Operations view shows the IBTrACS best track on top).
const fmt = (x: any) => (typeof x === "number" ? x.toFixed(0) : "–");

export default function RealEvents({ open }: { open: (c: string) => void }) {
  const [ev, setEv] = useState<RealEvent[] | null>(null);
  useEffect(() => { getEvents().then(setEv).catch(() => setEv([])); }, []);
  if (!ev) return <Skeleton h={300} label="Loading real events" />;
  return (
    <section aria-label="Real events view">
      <p className="note"><span className="badge real">REAL</span> ERA5 reanalysis and public ensemble forecasts against IBTrACS / IMD.
        REAL-TEST events are locked for the final blind evaluation (BRIEF4 Phase 8): no scores are shown for them before it.</p>
      <div className="card tablewrap">
        <table>
          <thead><tr><th scope="col">event</th><th scope="col">hazard</th><th scope="col">split</th>
            <th scope="col">GNN (temporal) mean track error, km</th><th scope="col">tracker v2, km</th><th scope="col">TE-style, km</th><th scope="col">dashboard</th></tr></thead>
          <tbody>{ev.map((e) => {
            const s = typeof e.scores === "object" && e.scores ? e.scores : null;
            const cyc = s && s.gnn_temporal;
            return (
              <tr key={e.event}>
                <td>{e.event}</td><td>{e.hazard.replace("_", " ")}</td><td>{e.split}</td>
                <td>{cyc ? `${fmt(s.gnn_temporal.err_mean_km)} (${s.gnn_temporal.matched_fixes}/${s.gnn_temporal.ibtracs_fixes} fixes)` : s?.methods ? `POD ${s.methods.gnn_temporal.pod.toFixed(2)}` : typeof e.scores === "string" ? e.scores : "–"}</td>
                <td>{cyc ? fmt(s.v2.err_mean_km) : s?.methods ? `POD ${s.methods.v2.pod.toFixed(2)}` : "–"}</td>
                <td>{cyc ? fmt(s.te_style.err_mean_km) : s?.methods ? `POD ${s.methods.te_style.pod.toFixed(2)}` : "–"}</td>
                <td>{e.case ? <button onClick={() => open(e.case!)}>open {e.case}</button> : "–"}</td>
              </tr>);
          })}</tbody>
        </table>
      </div>
      <p className="note">Numbers from reports/real_results.json (REAL_RESULTS.md). Heat/cold waves are scored against IMD criteria applied to ERA5 (not an independent truth).</p>
    </section>
  );
}
