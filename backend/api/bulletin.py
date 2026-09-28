"""IMD-style district bulletin as printable HTML, English or Hindi (BRIEF4 Phase 6).

Only fixed UI phrases are translated (backend/i18n/{en,hi}.json). District and state names are
kept exactly as GADM writes them, numbers are kept as digits, and the probability reason string
(an equation such as P(wind >= 118 km/h) = 0.72) is shown as is, so nothing is machine-translated.
"""
import html
import json
from pathlib import Path

I18N = Path(__file__).resolve().parents[1] / "i18n"
COL = {"low": "#FFD400", "moderate": "#FF8C00", "severe": "#D7191C"}


def strings(lang):
    return json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))


def render(meta, rollup, lang="en", lead=None):
    s = strings(lang)
    e = html.escape
    badge = s["synthetic"] if meta["synthetic"] else s["real"]
    rows = "".join(
        f"<tr><td>{e(r['district'])}</td><td>{e(r['state'])}</td>"
        f"<td><span class='lvl' style='background:{COL[r['category']]}'>{e(s['levels'][r['category']])}</span></td>"
        f"<td>{e(s['kinds'].get(r['kind'], r['kind']))}</td><td>{r['probability']:.2f}</td>"
        f"<td>{100 * r['fraction_of_district']:.0f}%</td><td>+{r['first_lead_h']}-{r['last_lead_h']}</td>"
        f"<td class='mono'>{e(r['reason'])}</td></tr>" for r in rollup["districts"])
    lead_txt = s["all_leads"] if lead is None else f"+{lead} h ({s['valid']} {e(meta['valid_times'][meta['leads_h'].index(lead)])} UTC)"
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(s['title'])} - {e(meta['case'])}</title>
<style>body{{font:14px/1.45 system-ui,'Noto Sans Devanagari','Nirmala UI',sans-serif;margin:24px;color:#111}}
h1{{font-size:20px;margin:0}} .sub{{color:#555}} .badge{{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;
background:{'#7b2fa8' if meta['synthetic'] else '#13663b'};font-weight:700}} table{{border-collapse:collapse;width:100%;margin-top:12px}}
th,td{{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}} th{{background:#f2f2f2}}
.lvl{{padding:1px 6px;border-radius:8px;font-weight:600}} .mono{{font-family:ui-monospace,monospace;font-size:12px}}
.note{{color:#555;font-size:12px}} @media print{{button{{display:none}}}}</style></head><body>
<h1>{e(s['title'])}</h1><div class="sub">{e(s['subtitle'])}</div>
<p><span class="badge">{e(badge)}</span> {e(s['case'])}: <b>{e(meta['case'])}</b> · {e(s['issued'])}: {e(meta['init_time'])} UTC · {e(s['lead'])}: {lead_txt}</p>
<button onclick="window.print()">{e(s['print'])}</button>
<table><thead><tr><th>{e(s['district'])}</th><th>{e(s['state'])}</th><th>{e(s['level'])}</th><th>{e(s['hazard'])}</th>
<th>{e(s['probability'])}</th><th>{e(s['coverage'])}</th><th>{e(s['leads'])}</th><th>{e(s['reason'])}</th></tr></thead>
<tbody>{rows or f"<tr><td colspan=8>{e(s['none'])}</td></tr>"}</tbody></table>
<p class="note">{e(s['districts_note'])}</p><p class="note">{e(s['disclaimer'])}</p></body></html>"""
