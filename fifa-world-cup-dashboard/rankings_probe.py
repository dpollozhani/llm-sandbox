#!/usr/bin/env python3
"""TEMPORARY probe: try FIFA ranking API variants and show what comes back."""

import json, urllib.request, urllib.error

ATTEMPTS = [
    ("inside.fifa.com", "https://inside.fifa.com/api/ranking-overview?locale=en"),
    ("www.fifa.com", "https://www.fifa.com/api/ranking-overview?locale=en"),
]
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fifa.com/en/rankings/men",
    "x-requested-with": "XMLHttpRequest",
}

for label, url in ATTEMPTS:
    print("=" * 60)
    print("GET", url)
    try:
        req = urllib.request.Request(url, headers=HDRS)
        with urllib.request.urlopen(req, timeout=60) as r:
            ct = r.headers.get("content-type")
            body = r.read().decode("utf-8", "ignore")
        print("status OK, content-type:", ct, "bytes:", len(body))
        s = body.lstrip()
        if s[:1] in "{[":
            data = json.loads(body)
            print("JSON top keys:", list(data.keys()) if isinstance(data, dict) else "(list)")
            rk = (data.get("rankings") if isinstance(data, dict) else None) or []
            print("rankings count:", len(rk))
            if rk:
                print("entry keys:", list(rk[0].keys()))
                print("sample:", json.dumps(rk[0])[:400])
        else:
            print("NOT JSON; first 300 chars:")
            print(body[:300])
    except urllib.error.HTTPError as e:
        print("HTTPError", e.code, e.reason, "body:", e.read()[:200])
    except Exception as e:
        print("ERR", repr(e))
