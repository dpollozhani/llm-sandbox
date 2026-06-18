#!/usr/bin/env python3
"""Scrape FIFA World Cup results into a flat JSON file.

Source: ESPN's public scoreboard JSON (no API key, stable shape). ESPN carries
the official World Cup fixtures/results and is far more reliable to scrape than
fifa.com (which is bot-protected). The whole source is isolated here, so it can
be swapped without touching the dashboard.

Output: data/matches.json next to this script. The dashboard reads that file
directly (same origin, so no CORS), and a GitHub Action keeps it fresh.

Stdlib only — no pip install needed in CI.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone

# ESPN soccer league slug for the FIFA World Cup.
LEAGUE = os.environ.get("WC_LEAGUE", "fifa.world")
# 2026 tournament window (inclusive). Override via env if dates shift.
START = os.environ.get("WC_START", "2026-06-11")
END = os.environ.get("WC_END", "2026-07-19")

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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(OUT_DIR, "matches.json")


def daterange(start_s, end_s):
    start = datetime.strptime(start_s, "%Y-%m-%d").date()
    end = datetime.strptime(end_s, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    end = min(end, today)  # don't bother fetching future days
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(d):
    url = SCOREBOARD.format(league=LEAGUE, d=d.strftime("%Y%m%d"))
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


def load_existing():
    """Return {id: match} from the current data file, if any."""
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
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


def main():
    scraped = {}
    errors = 0
    days = 0
    for d in daterange(START, END):
        days += 1
        try:
            payload = fetch_day(d)
            for m in parse_events(payload):
                if m["id"]:
                    scraped[m["id"]] = m
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            errors += 1
            print(f"  ! {d}: {e}", file=sys.stderr)
        time.sleep(0.3)  # be polite

    existing = load_existing()
    merged = merge(existing, scraped)

    # Nothing anywhere (first run that fetched nothing): fail loudly, keep file as-is.
    if not merged:
        print("No data from scrape and no existing file; leaving things untouched.", file=sys.stderr)
        sys.exit(1)

    ordered = sorted(merged.values(), key=lambda m: (m.get("date") or "", m.get("id")))
    finished = [m for m in ordered if m["completed"]
                and m["homeScore"] is not None and m["awayScore"] is not None]

    # Skip rewriting when the match data is unchanged, so we don't churn commits
    # (and the "updated" timestamp stays meaningful: when data actually changed).
    if merged == existing:
        print(f"No changes (merged total {len(ordered)}, {len(finished)} finished, "
              f"{errors} day(s) errored). Data file left untouched.")
        return

    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ESPN (site.api.espn.com)",
        "league": LEAGUE,
        "window": {"start": START, "end": END},
        "totalMatches": len(ordered),
        "finishedMatches": len(finished),
        "matches": ordered,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated {OUT_FILE}: scraped {len(scraped)} across {days} day(s), "
          f"merged total {len(ordered)} ({len(finished)} finished), {errors} day(s) errored.")


if __name__ == "__main__":
    main()
