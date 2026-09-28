"""Fetch every URL in research/PRIOR_WORK.md and record the HTTP status.

http_status is the FINAL status after following redirects (a doi.org 302 that
lands on a 403 publisher page counts as 403). first_hop_status is also kept.
With --prune, table rows whose link is not 2xx/3xx are removed from PRIOR_WORK.md.
"""
import csv
import datetime as dt
import re
import sys
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "research" / "PRIOR_WORK.md"
CSV = ROOT / "research" / "LINKS_CHECK.csv"
URL_RE = re.compile(r"https?://[^\s|)<>\]]+")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9"}


def check(url):
    last = None
    for attempt in range(2):
        try:
            first = requests.get(url, headers=UA, timeout=30, allow_redirects=False)
            r = requests.get(url, headers=UA, timeout=30, allow_redirects=True)
            return first.status_code, r.status_code, r.url
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(2)
    return None, last, ""


def good(status):
    return isinstance(status, int) and 200 <= status < 400


def main(prune):
    text = MD.read_text(encoding="utf-8")
    urls = list(dict.fromkeys(u.rstrip(".,;") for u in URL_RE.findall(text)))
    rows = []
    for u in urls:
        fh, st, final = check(u)
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rows.append({"url": u, "http_status": st, "checked_at": now,
                     "first_hop_status": fh, "final_url": final})
        print(f"{st}\t{u}")
    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r["url"] for r in rows if not good(r["http_status"])]
    print(f"GOOD={len(rows)-len(bad)} BAD={len(bad)}")
    if prune and bad:
        kept = [ln for ln in text.splitlines(keepends=True)
                if not (ln.lstrip().startswith("|") and any(b in ln for b in bad))]
        MD.write_text("".join(kept), encoding="utf-8")
        # drop bad rows from the csv too, so it reflects the final file
        with CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([r for r in rows if good(r["http_status"])])
        print("pruned:", *bad, sep="\n  ")


if __name__ == "__main__":
    main("--prune" in sys.argv)
