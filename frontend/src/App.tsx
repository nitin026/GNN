import { lazy, Suspense, useEffect, useState } from "react";
import { detectMode, getCases, getMeta, type Mode } from "./api";
import { Badge, Skeleton } from "./components/ui";
import type { CaseSummary, Meta } from "./types";

const Operations = lazy(() => import("./views/Operations"));
const Downscaling = lazy(() => import("./views/Downscaling"));
const Alerts = lazy(() => import("./views/Alerts"));
const Performance = lazy(() => import("./views/Performance"));
const Forecaster = lazy(() => import("./views/Forecaster"));
const RealEvents = lazy(() => import("./views/RealEvents"));
const VIEWS = [["ops", "Operations"], ["fc", "Forecaster"], ["ds", "Downscaling"], ["alerts", "Alerts"], ["real", "Real events"], ["perf", "Model performance"]] as const;
type View = (typeof VIEWS)[number][0];

function initial<T extends string>(key: string, fallback: T): T {
  const q = new URLSearchParams(location.search).get(key);
  if (q) return q as T;
  try { return (localStorage.getItem(`sih.${key}`) as T) || fallback; } catch { return fallback; }
}
function remember(key: string, v: string) { try { localStorage.setItem(`sih.${key}`, v); } catch { /* private mode */ } }

export default function App() {
  const [mode, setMode] = useState<Mode | null>(null);
  const [cases, setCases] = useState<CaseSummary[] | null>(null);
  const [caseId, setCaseId] = useState<string>(initial("case", "amphan_replay"));
  const [meta, setMeta] = useState<Meta | null>(null);
  const [lead, setLead] = useState<number>(Number(new URLSearchParams(location.search).get("lead") ?? 0));
  const [view, setView] = useState<View>(initial<View>("view", "ops"));
  const [theme, setTheme] = useState<"dark" | "light">(initial("theme", "dark"));
  const [error, setError] = useState("");

  useEffect(() => { document.documentElement.dataset.theme = theme; remember("theme", theme); }, [theme]);
  useEffect(() => { remember("view", view); }, [view]);
  useEffect(() => {
    detectMode().then((m) => { setMode(m); return getCases(); }).then((cs) => {
      setCases(cs);
      if (!cs.some((c) => c.case === caseId) && cs.length) setCaseId(cs[0].case);
    }).catch((e) => setError(String(e)));
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!mode || !cases) return;
    setMeta(null);
    remember("case", caseId);
    getMeta(caseId).then((m) => { setMeta(m); setLead((l) => (m.leads_h.includes(l) ? l : 0)); }).catch((e) => setError(String(e)));
  }, [caseId, mode, cases]);

  return (
    <div className="app">
      <header>
        <h1>Extreme-weather anomaly tracking <span className="sub">NEPS-G style ensembles · 12→5 km</span></h1>
        <label>Case <select value={caseId} onChange={(e) => setCaseId(e.target.value)} aria-label="Case">
          {(cases ?? []).map((c) => <option key={c.case} value={c.case}>{c.case} · {c.hazard.replace("_", " ")} · {c.badge}</option>)}
        </select></label>
        {meta && <Badge synthetic={meta.synthetic} text={meta.badge} />}
        <span className={`mode ${mode}`} title={mode === "api" ? "Live REST API" : "API unreachable: reading precomputed files"}>{mode === "api" ? "● API" : mode ? "○ offline demo" : "…"}</span>
        <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle light/dark theme">{theme === "dark" ? "☀ light" : "☾ dark"}</button>
      </header>
      <nav role="tablist" aria-label="Views">
        {VIEWS.map(([k, n]) => <button key={k} role="tab" aria-selected={view === k} className={view === k ? "on" : ""} onClick={() => setView(k)}>{n}</button>)}
      </nav>
      {meta && <p className="source">{meta.synthetic ? "SYNTHETIC event with exact ground truth on a real ERA5 background — not an observation or forecast. " : ""}Source: {meta.source}</p>}
      <main>
        {error && <p className="err" role="alert">{error}</p>}
        {!meta ? <Skeleton h={560} label="Loading case" /> :
          <Suspense fallback={<Skeleton h={560} />}>
            {view === "ops" && <Operations meta={meta} lead={lead} setLead={setLead} theme={theme} />}
            {view === "ds" && <Downscaling meta={meta} lead={lead} />}
            {view === "alerts" && <Alerts meta={meta} lead={lead} setLead={setLead} theme={theme} />}
            {view === "perf" && <Performance />}
            {view === "fc" && <Forecaster meta={meta} lead={lead} setLead={setLead} />}
            {view === "real" && <RealEvents open={(c) => { setCaseId(c); setView("ops"); }} />}
          </Suspense>}
      </main>
      <footer className="note">SIH PS 26078 (MoES / NCMRWF) research prototype. Not for operational use without NEPS-G validation. Basemap © OpenStreetMap contributors.</footer>
    </div>
  );
}
