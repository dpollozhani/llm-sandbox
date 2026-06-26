#!/usr/bin/env python3
"""TEMPORARY probe: dump Wikipedia tables that mention FIFA ranking, plus our
ESPN team names, so we can design the real rankings parser + alias map.
Runs in CI (workflow_dispatch). Safe to delete afterwards."""

import json, re, html, os, urllib.request

PAGES = {"2018": "2018_FIFA_World_Cup", "2022": "2022_FIFA_World_Cup", "2026": "2026_FIFA_World_Cup"}
HDRS = {"User-Agent": "wc-dashboard-probe/1.0 (github actions)"}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def cells(row):
    return [strip(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S | re.I)]


for ed, page in PAGES.items():
    print("=" * 70)
    print("EDITION", ed, "->", page)
    f = "matches.json" if ed == "2026" else ed + ".json"
    try:
        d = json.load(open(os.path.join(DATA, f)))
        names = sorted(set([m["home"] for m in d["matches"]] + [m["away"] for m in d["matches"]]))
        print("ESPN teams (%d): %s" % (len(names), ", ".join(names)))
    except Exception as e:
        print("ESPN load err:", e)
    try:
        h = fetch("https://en.wikipedia.org/api/rest_v1/page/html/" + page)
    except Exception as e:
        print("wiki fetch err:", e)
        continue
    for ti, t in enumerate(re.findall(r"<table[^>]*>(.*?)</table>", h, re.S | re.I)):
        if re.search(r"FIFA", t, re.I) and re.search(r"rank", t, re.I):
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S | re.I)
            if len(rows) < 6:
                continue
            print("---- candidate table #%d (%d rows) ----" % (ti, len(rows)))
            for r in rows[:7]:
                c = cells(r)
                if c:
                    print(" | ".join(c)[:220])
