import { useEffect, useMemo, useState } from "react";
import MapView from "../components/MapView";
import { LeadSlider, SEV } from "../components/ui";
import { alertsAt, currentMode, getAlerts, getDistricts, staticUrl } from "../api";
import * as L from "../layers";
import type { Alert, DistrictRow, Meta } from "../types";

export default function Alerts({ meta, lead, setLead, theme }: { meta: Meta; lead: number; setLead: (l: number) => void; theme: "dark" | "light" }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [sel, setSel] = useState<Alert | null>(null);
  const [allLeads, setAllLeads] = useState(false);
  const [districts, setDistricts] = useState<DistrictRow[] | null>(null);
  const [loc, setLoc] = useState({ lat: "21.5", lon: "87.0" });
  const [mine, setMine] = useState<{ alerts: Alert[]; exact: boolean } | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => { getAlerts(meta.case, allLeads ? undefined : lead).then((a) => { setAlerts(a); setSel(null); }); }, [meta.case, lead, allLeads]);
  useEffect(() => { getDistricts(meta.case, allLeads ? undefined : lead).then(setDistricts); }, [meta.case, lead, allLeads]);

  const check = async (lat = Number(loc.lat), lon = Number(loc.lon)) => {
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < 0 || lat > 40 || lon < 60 || lon > 100) {
      setErr("Enter a point inside the India box (0–40° N, 60–100° E)."); return;
    }
    setErr("");
    setLoc({ lat: lat.toFixed(2), lon: lon.toFixed(2) });
    setMine(await alertsAt(meta.case, lat, lon, allLeads ? undefined : lead));
  };

  const layers = useMemo(() => [
    L.outline("districts", staticUrl("districts.geojson"), [150, 160, 170, 90]),
    L.outline("india", staticUrl("india_outline.geojson")),
    ...L.alertPins(alerts, sel?.id, setSel),
  ], [alerts, sel]);

  const counts = (["severe", "moderate", "low"] as const).map((c) => [c, alerts.filter((a) => a.category === c).length] as const);
  return (
    <section aria-label="Alerts view">
      <div className="toolbar">
        <LeadSlider lead={lead} leads={meta.leads_h} onChange={setLead} validTime={meta.valid_times[meta.leads_h.indexOf(lead)]} />
        <label className="chk"><input type="checkbox" checked={allLeads} onChange={() => setAllLeads(!allLeads)} />all leads</label>
        <span className="counts">{counts.map(([c, n]) => <span key={c} className="pill" style={{ background: SEV[c] }}>{n} {c}</span>)}</span>
      </div>
      <div className="grid2">
        <div>
          <MapView layers={layers} theme={theme} bounds={[62, 2, 98, 36]} label="Alert map: pins with 5 km impact radius, coloured by IMD severity"
            getTooltip={L.tooltip} onClick={(lat, lon) => check(lat, lon)} />
          <form className="card locate" onSubmit={(e) => { e.preventDefault(); check(); }}>
            <h2>Check my location</h2>
            <label>lat <input inputMode="decimal" value={loc.lat} onChange={(e) => setLoc({ ...loc, lat: e.target.value })} aria-label="latitude" /></label>
            <label>lon <input inputMode="decimal" value={loc.lon} onChange={(e) => setLoc({ ...loc, lon: e.target.value })} aria-label="longitude" /></label>
            <button type="submit">Check</button>
            <span className="note">or click the map</span>
            {err && <p className="err" role="alert">{err}</p>}
            {mine && <div aria-live="polite">
              <p><b>{mine.alerts.length ? `${mine.alerts.length} alert(s) cover this point` : "No alert covers this point"}</b>
                {!mine.exact && <span className="note"> (offline mode: alert-region bounding box, not the exact region)</span>}</p>
              <ul className="plain">{mine.alerts.slice(0, 8).map((a) => <li key={a.id}><span className="pill" style={{ background: a.colour }}>{a.category}</span> {a.reason}, +{a.lead_h} h
                {a.distance_to_pinpoint_km !== undefined && <> · {a.distance_to_pinpoint_km} km from core</>}</li>)}</ul>
            </div>}
          </form>
        </div>
        <div className="card">
          <h2>Alerts ({allLeads ? "all leads" : `+${lead} h`}), most severe first</h2>
          {sel && <div className="explain" style={{ borderColor: sel.colour }} aria-live="polite">
            <b>{sel.category.toUpperCase()} · {sel.kind}</b> {sel.in_india ? "over India" : "marine / outside India"}
            <div>{sel.reason}</div>
            <div className="note">Ensemble probability {sel.probability} ({sel.probability_type}); per-cell {sel.probability_cell}; {sel.n_members} members.
              Core {sel.pinpoint.lat.toFixed(2)}° N {sel.pinpoint.lon.toFixed(2)}° E ({sel.pinpoint.grid}), 5 km impact radius, valid {sel.valid_time} UTC.
              Region {sel.region_cells_12km} cells at 12 km. {meta.synthetic ? "SYNTHETIC case." : ""}</div>
          </div>}
          <div className="tablewrap">
            <table>
              <thead><tr><th scope="col">severity</th><th scope="col">kind</th><th scope="col">lead</th><th scope="col">P</th><th scope="col">core</th></tr></thead>
              <tbody>{alerts.slice(0, 300).map((a) => (
                <tr key={a.id} className={sel?.id === a.id ? "sel" : ""} tabIndex={0} onClick={() => setSel(a)} onKeyDown={(e) => e.key === "Enter" && setSel(a)}>
                  <td><span className="pill" style={{ background: a.colour }}>{a.category}</span></td><td>{a.kind}</td><td>+{a.lead_h} h</td>
                  <td>{a.probability.toFixed(2)}</td><td>{a.pinpoint.lat.toFixed(2)}, {a.pinpoint.lon.toFixed(2)}</td></tr>))}</tbody>
            </table>
            {!alerts.length && <p className="note">No alerts{allLeads ? "" : " at this lead"}.</p>}
          </div>
          <h2>District bulletin</h2>
          {districts === null ? <p className="note">{currentMode() === "api" ? "No district roll-up for this case." : "Offline mode: the district roll-up needs the API (python -m backend.api)."}</p> :
            <div className="tablewrap"><table>
              <thead><tr><th scope="col">district</th><th scope="col">state</th><th scope="col">severity</th><th scope="col">kind</th><th scope="col">P</th><th scope="col">leads</th></tr></thead>
              <tbody>{districts.slice(0, 200).map((d) => (
                <tr key={d.gid + d.kind}><td>{d.district}</td><td>{d.state}</td><td><span className="pill" style={{ background: d.colour }}>{d.category}</span></td>
                  <td>{d.kind}</td><td title={d.reason}>{d.probability.toFixed(2)}</td><td>+{d.first_lead_h}–{d.last_lead_h} h</td></tr>))}</tbody>
            </table>{!districts.length && <p className="note">No district is warned.</p>}</div>}
          <p className="note">IMD colours: yellow = low, orange = moderate, red = severe. Categories = ensemble probability × IMD thresholds (docs/ALERT_RULES.md). Districts: GADM 4.1 level 2.</p>
        </div>
      </div>
    </section>
  );
}
