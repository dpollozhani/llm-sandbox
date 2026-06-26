#!/usr/bin/env python3
"""TEMPORARY probe: extract the dateId<->date map from __NEXT_DATA__ and find
the releases for the 2018/2022/2026 World Cups."""

import json, re, urllib.request

HDRS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "*/*"}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


page = fetch("https://inside.fifa.com/fifa-world-ranking/men")
m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
print("found __NEXT_DATA__:", bool(m))
data = json.loads(m.group(1))

# Collect every dict that carries an id-token AND an ISO date.
seen = set()
hits = []
def walk(o):
    if isinstance(o, dict):
        blob = json.dumps(o, ensure_ascii=False)
        if re.search(r'\bid\d{3,6}\b', blob) and re.search(r'20\d\d-\d\d-\d\d', blob) and len(blob) < 400:
            key = blob
            if key not in seen:
                seen.add(key); hits.append(o)
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(data)
print("date-ish dicts:", len(hits))

want = re.compile(r'2018-0[5-7]|2022-(09|10|11)|2026-0[5-7]')
print("\n== entries near WC releases ==")
for h in hits:
    s = json.dumps(h, ensure_ascii=False)
    if want.search(s):
        print(s[:240])

print("\n== sample of all date-ish dicts (first 8) ==")
for h in hits[:8]:
    print(json.dumps(h, ensure_ascii=False)[:200])
