#!/usr/bin/env python3
"""TEMPORARY probe: inspect FIFA's hidden ranking API so we can build the real
parser. Dumps top-level keys, the list of available ranking dates (to find the
June 2026 release), and a few sample ranking entries with their field names.
Runs in CI (workflow_dispatch). Safe to delete afterwards."""

import json, urllib.request

HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


URL = "https://www.fifa.com/api/ranking-overview?locale=en"
print("GET", URL)
try:
    raw = fetch(URL)
    print("bytes:", len(raw))
    data = json.loads(raw)
    print("TOP-LEVEL KEYS:", list(data.keys()))

    # Look for the available ranking-release dates (to find June/July 2026).
    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(
            "date" in (kk.lower()) or kk.lower() in ("id", "iso", "text") for kk in v[0]
        ) and k != "rankings":
            print(f"\nPOSSIBLE DATES FIELD '{k}' ({len(v)} entries); last 8:")
            for d in v[-8:]:
                print("  ", json.dumps(d)[:200])

    rk = data.get("rankings") or []
    print("\nrankings count:", len(rk))
    if rk:
        print("ENTRY KEYS:", list(rk[0].keys()))
        ri = rk[0].get("rankingItem")
        if isinstance(ri, dict):
            print("rankingItem KEYS:", list(ri.keys()))
        print("\nTOP 6 ENTRIES (trimmed):")
        for e in rk[:6]:
            print("  ", json.dumps(e)[:300])
except Exception as ex:
    print("ERROR:", repr(ex))
