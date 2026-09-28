import { useEffect, useMemo, useState } from "react";
import MapView from "../components/MapView";
import { Legend, LeadSlider, Skeleton } from "../components/ui";
import { cutTracks, fieldUrl, getAlerts, getTracks, staticUrl } from "../api";
import * as L from "../layers";
import type { Alert, FeatureCollection, Meta } from "../types";

const VARS = [["anom", "Anomaly (z)"], ["msl", "MSLP"], ["wind", "10 m wind"], ["tp", "6 h rain"], ["t2m", "2 m temperature"]] as const;

export default function Operations({ meta, lead, setLead, theme }: { meta: Meta; lead: number; setLead: (l: number) => void; theme: "dark" | "light" }) {
  const [v, setV] = useState<string>("anom");
  const [show, setShow] = useState({ field: true, strike: true, tracks: true, cone: true, box: true, alerts: true, box3d: false });
  const [tracks, setTracks] = useState<FeatureCollection | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  useEffect(() => { setTracks(null); getTracks(meta.case).then(setTracks); }, [meta.case]);
  useEffect(() => { getAlerts(meta.case, lead).then(setAlerts); }, [meta.case, lead]);
  const cut = useMemo(() => (tracks ? cutTracks(tracks, lead) : null), [tracks, lead]);

  const layers = useMemo(() => {
    const out: any[] = [];
    if (show.field) out.push(L.field(`f-${v}-${lead}`, fieldUrl(meta.case, v, "12km", lead), meta, 0.8));
    if (show.strike) out.push(L.field(`s-${lead}`, fieldUrl(meta.case, "strike", "12km", lead), meta, 0.75));
    out.push(L.outline("india", staticUrl("india_outline.geojson")));
    if (cut) {
      if (show.tracks) out.push(L.memberTracks(cut));
      if (show.cone) out.push(...L.consensus(cut));
      if (show.box) out.push(L.box4d(cut, show.box3d));
    }
    if (show.alerts) out.push(...L.alertPins(alerts));
    return out;
  }, [meta, v, lead, show, cut, alerts]);

  const toggle = (k: keyof typeof show) => setShow({ ...show, [k]: !show[k] });
  const ti = meta.leads_h.indexOf(lead);
  return (
    <section aria-label="Operations view">
      <div className="toolbar">
        <LeadSlider lead={lead} leads={meta.leads_h} onChange={setLead} validTime={meta.valid_times[ti]} />
        <label>Field <select value={v} onChange={(e) => setV(e.target.value)}>{VARS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
        {([["field", "field"], ["strike", "strike probability"], ["tracks", "ensemble tracks"], ["cone", "consensus + cone"], ["box", "4-D box"], ["box3d", "3-D (time-extruded)"], ["alerts", "alerts"]] as const).map(([k, n]) =>
          <label key={k} className="chk"><input type="checkbox" checked={show[k]} onChange={() => toggle(k)} />{n}</label>)}
      </div>
      {!tracks ? <Skeleton h={520} label="Loading tracks" /> :
        <MapView layers={layers} theme={theme} pitch={show.box3d ? 55 : 0} bounds={[62, 2, 98, 36]} getTooltip={L.tooltip} label="Operations map: anomaly field, strike probability, tracks, 4-D box and alerts" />}
      <div className="legends">
        {show.field && <Legend legend={meta.legends[v]} title={VARS.find(([k]) => k === v)![1]} />}
        {show.strike && <Legend legend={meta.legends.strike} title="Strike probability" />}
        <span className="legend"><i className="sw" style={{ background: "#56B4E9" }} />consensus + spread cone (km)</span>
        <span className="legend"><i className="sw" style={{ background: "#F0E442" }} />4-D box{show.box3d ? " (height = lead time, 2.5 km per h)" : ""}</span>
      </div>
      <p className="note">Field panel member: {meta.fields_member}. Tracker: {meta.tracker}. {alerts.length} alert(s) at this lead.</p>
    </section>
  );
}
