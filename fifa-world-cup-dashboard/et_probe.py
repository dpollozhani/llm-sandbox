#!/usr/bin/env python3
"""TEMPORARY probe: can we recover the score-at-90' for ET/penalty games?

Checks two ESPN endpoints for a few known extra-time / penalty matches:
  - scoreboard competitor `linescores` (per-period goals), and
  - summary `keyEvents` / `scoringPlays` (per-goal period + clock).
Prints enough structure to decide how to derive the regulation (<=90') score.
Run in CI only (this sandbox can't reach ESPN). Deleted after validation.
"""
import json
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={d}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={id}"

# (event id, YYYYMMDD, label) — known ET/pens games
TARGETS = [
    ("498141", "20180711", "Croatia 2-1 England (AET; 1-1 at 90')"),
    ("498152", "20180701", "Spain 1-1 Russia (pens)"),
    ("633850", "20221218", "Argentina 3-3 France (pens; 2-2 at 90')"),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def probe_scoreboard(eid, date):
    print(f"  -- scoreboard {date} --")
    try:
        pay = get(SCOREBOARD.format(d=date))
    except Exception as e:
        print("    scoreboard error:", e); return
    for ev in pay.get("events", []):
        if str(ev.get("id")) != eid:
            continue
        comp = (ev.get("competitions") or [{}])[0]
        for c in comp.get("competitors", []):
            ls = c.get("linescores")
            print(f"    {c.get('homeAway')}: score={c.get('score')} "
                  f"linescores={ls} shootoutScore={c.get('shootoutScore')}")
        st = (comp.get("status") or {}).get("type") or {}
        print(f"    status: detail={st.get('detail')!r} period={comp.get('status',{}).get('period')}")


def probe_summary(eid):
    print(f"  -- summary event={eid} --")
    try:
        pay = get(SUMMARY.format(id=eid))
    except Exception as e:
        print("    summary error:", e); return
    print("    top-level keys:", sorted(pay.keys()))
    # team id -> home/away from header
    idmap = {}
    hdr = pay.get("header") or {}
    for c in ((hdr.get("competitions") or [{}])[0]).get("competitors", []):
        idmap[str((c.get("team") or {}).get("id"))] = (c.get("homeAway"), c.get("score"))
    print("    header competitors:", idmap)

    for field in ("keyEvents", "scoringPlays"):
        evs = pay.get(field)
        if not evs:
            print(f"    {field}: (absent/empty)")
            continue
        print(f"    {field}: {len(evs)} entries")
        reg = {"home": 0, "away": 0}
        for e in evs:
            sp = e.get("scoringPlay")
            typ = (e.get("type") or {}).get("text") or (e.get("type") or {}).get("id")
            per = (e.get("period") or {}).get("number")
            clk = (e.get("clock") or {}).get("displayValue")
            tid = str((e.get("team") or {}).get("id"))
            ha = idmap.get(tid, ("?", "?"))[0]
            own = e.get("ownGoal")
            print(f"      sp={sp} type={typ!r} period={per} clock={clk} team={ha} own={own}")
            if sp and per and per <= 2:
                side = "away" if (ha == "home") == bool(own) else "home"  # own goal credits other side
                if ha in ("home", "away"):
                    reg[side] += 1
        print(f"    -> derived <=90' (period<=2) goals: {reg}")


def main():
    for eid, date, label in TARGETS:
        print(f"\n==== {label}  (id {eid}) ====")
        probe_scoreboard(eid, date)
        probe_summary(eid)


if __name__ == "__main__":
    main()
