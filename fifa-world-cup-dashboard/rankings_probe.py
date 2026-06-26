#!/usr/bin/env python3
"""TEMPORARY probe: find FIFA ranking dateIds, then pull one release to learn
the entry shape. inside.fifa.com returns clean JSON."""

import json, re, urllib.request

HDRS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "*/*"}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


# 1) Pull the ranking landing page and look for dateIds embedded in __NEXT_DATA__.
page = fetch("https://inside.fifa.com/fifa-world-ranking/men")
print("page bytes:", len(page))
ids = sorted(set(re.findall(r'id\d{3,6}', page)))
print("dateId-looking tokens found:", ids[:40])

# 2) If we found any, call the API with the first and dump the dates list + a sample.
if ids:
    did = ids[-1]
    url = "https://inside.fifa.com/api/ranking-overview?locale=en&dateId=" + did
    print("\nGET", url)
    data = json.loads(fetch(url))
    print("top keys:", list(data.keys()))
    dates = data.get("dates") or []
    print("dates count:", len(dates))
    for d in dates:
        s = json.dumps(d)
        if re.search(r'2018-0[67]|2022-1[01]|2026-0[67]', s):
            print("  MATCH:", s[:200])
    rk = data.get("rankings") or []
    print("rankings count:", len(rk))
    if rk:
        print("entry sample:", json.dumps(rk[0])[:500])
