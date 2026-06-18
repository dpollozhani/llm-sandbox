#!/usr/bin/env python3
"""Scrape FIFA World Cup results into flat JSON files (one per tournament).

Source: ESPN's public scoreboard JSON (no API key, stable shape). ESPN carries
the official World Cup fixtures/results and is far more reliable to scrape than
fifa.com (which is bot-protected). The whole source is isolated here, so it can
be swapped without touching the dashboard.

Output: data/<edition>.json next to this script (e.g. data/matches.json for
2026, data/2022.json, data/2018.json). The dashboard reads these directly
(same origin, so no CORS), and a GitHub Action keeps the current one fresh.

The current edition is scraped on every run and merged incrementally so it can
only grow/refine (never blanks out). Finished historical editions are scraped
once (when their file is missing/empty) and then skipped, unless WC_FORCE=1.

Stdlib only — no pip install needed in CI.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

# Each World Cup edition: ESPN league slug, tournament window, and output file.
EDITIONS = {
    "2026": {"league": "fifa.world", "start": "2026-06-11", "end": "2026-07-19", "file": "matches.json"},
    "2022": {"league": "fifa.world", "start": "2022-11-20", "end": "2022-12-18", "file": "2022.json"},
    "2018": {"league": "fifa.world", "start": "2018-06-14", "end": "2018-07-15", "file": "2018.json"},
}

FORCE = os.environ.get("WC_FORCE") == "1"

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "{league}/scoreboard?dates={d}"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def daterange(start_s, end_s):
    start = datetime.strptime(start_s, "%Y-%m-%d").date()
    end = datetime.strptime(end_s, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    end = min(end, today)  # don't bother fetching future days
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(league, d):
    url = SCOREBOARD.format(league=league, d=d.strftime("%Y%m%d"))
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def parse_events(payload):
    """Yield normalized match dicts from one scoreboard payload."""
    for ev in (payload or {}).get("events", []) or []:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        status = (comp.get("status") or ev.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed"))

        def team_name(c):
            t = c.get("team") or {}
            return t.get("displayName") or t.get("name") or t.get("shortDisplayName") or "?"

        def team_abbr(c):
            t = c.get("team") or {}
            return t.get("abbreviation") or (team_name(c)[:3].upper())

        yield {
            "id": str(ev.get("id") or comp.get("id") or ""),
            "date": ev.get("date") or comp.get("date") or "",
            "home": team_name(home),
            "away": team_name(away),
            "homeAbbr": team_abbr(home),
            "awayAbbr": team_abbr(away),
            "homeScore": to_int(home.get("score")),
            "awayScore": to_int(away.get("score")),
            "completed": completed,
            "status": status.get("description") or status.get("detail") or status.get("state") or "",
        }


def load_existing(path):
    """Return {id: match} from an existing data file, if any."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {m["id"]: m for m in data.get("matches", []) if m.get("id")}
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def merge(existing, scraped):
    """Incrementally fold scraped matches into the existing ones.

    Guarantees the dataset can only grow or refine, never shrink:
    - a previously known match is never dropped (so a transient empty/failed
      API response can't blank the file);
    - a finished result is never downgraded back to a non-finished one.
    """
    result = dict(existing)
    for mid, m in scraped.items():
        old = result.get(mid)
        if old is None or m["completed"] or not old.get("completed"):
            result[mid] = m
    return result


def scrape_edition(key, cfg):
    """Scrape one edition and write its file if the data changed."""
    path = os.path.join(DATA_DIR, cfg["file"])
    existing = load_existing(path)
    end = datetime.strptime(cfg["end"], "%Y-%m-%d").date()
    is_current = end >= datetime.now(timezone.utc).date()

    # Skip finished editions we already captured (re-scrape only with WC_FORCE=1).
    if not is_current and existing and not FORCE:
        print(f"[{key}] already captured ({len(existing)} matches), skipping.")
        return

    scraped = {}
    errors = days = 0
    for d in daterange(cfg["start"], cfg["end"]):
        days += 1
        try:
            for m in parse_events(fetch_day(cfg["league"], d)):
                if m["id"]:
                    scraped[m["id"]] = m
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            errors += 1
            print(f"  ! [{key}] {d}: {e}", file=sys.stderr)
        time.sleep(0.3)  # be polite

    merged = merge(existing, scraped)
    if not merged:
        print(f"[{key}] no data from scrape and no existing file.", file=sys.stderr)
        return False

    ordered = sorted(merged.values(), key=lambda m: (m.get("date") or "", m.get("id")))
    finished = [m for m in ordered if m["completed"]
                and m["homeScore"] is not None and m["awayScore"] is not None]

    if merged == existing:
        print(f"[{key}] no changes ({len(ordered)} matches, {len(finished)} finished).")
        return True

    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ESPN (site.api.espn.com)",
        "edition": key,
        "league": cfg["league"],
        "window": {"start": cfg["start"], "end": cfg["end"]},
        "totalMatches": len(ordered),
        "finishedMatches": len(finished),
        "matches": ordered,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[{key}] wrote {cfg['file']}: scraped {len(scraped)} across {days} day(s), "
          f"merged total {len(ordered)} ({len(finished)} finished), {errors} day(s) errored.")
    return True


def main():
    results = [scrape_edition(k, cfg) for k, cfg in EDITIONS.items()]
    # Fail the run only if a current edition produced nothing at all.
    if any(r is False for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
