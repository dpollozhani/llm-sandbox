#!/usr/bin/env python3
"""TEMPORARY probe: dump the whereig FIFA-rankings table structure + our ESPN
2026 team names, to build the parser + alias map. Safe to delete after."""

import json, re, html, os, urllib.request

HDRS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml"}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def cells(row):
    return [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]


page = fetch("https://www.whereig.com/football/fifa-world-rankings.html")
print("page bytes:", len(page))
tables = re.findall(r"<table[^>]*>(.*?)</table>", page, re.S | re.I)
print("tables found:", len(tables))
if tables:
    best = max(tables, key=lambda t: len(re.findall(r"<tr", t, re.I)))
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", best, re.S | re.I)
    print("rows in biggest table:", len(rows))
    print("---- first 16 rows ----")
    for r in rows[:16]:
        c = cells(r)
        if c:
            print(" | ".join(c)[:160])

f = os.path.join(DATA, "matches.json")
try:
    d = json.load(open(f))
    names = sorted(set([m["home"] for m in d["matches"]] + [m["away"] for m in d["matches"]]))
    print("\nESPN 2026 teams (%d): %s" % (len(names), ", ".join(names)))
except Exception as e:
    print("ESPN load err:", e)
