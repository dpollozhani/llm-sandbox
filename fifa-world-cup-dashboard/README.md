# FIFA World Cup Dashboard ⚽

A mobile-first dashboard showing the **accumulated results of the FIFA World Cup**:
goals-per-game distribution, distinct-scoreline frequency, summary stats, and a
results list. Built to open on your phone. A selector switches between the
**2026, 2022, and 2018** tournaments.

Controls: pick a **tournament** (2026 / 2022 / 2018, or **All-time** combined),
filter by **stage** (group / knockout), toggle the charts between **count** and
**percent** (relative frequency), and use the **completion slider** to see the
distribution at any point in the tournament (e.g. "50% of matches played"). The
completion cutoff is applied per edition, so it's comparable across tournaments and
works in All-time too. The All-time view is always relative, since the tournaments
have different numbers of matches. Both charts carry a cumulative **Pareto** line and
an **80%** guide, so you can read how concentrated outcomes are ("80% of games have
≤ 4 goals", "80% of results are the 8 most common").

The dashboard is built around one question: *how predictable is a game, and what does
each piece of information buy you?* Two controls drive that:

- **Split by** — view the distributions **combined**, **by ranking gap** (small-multiple
  Pareto charts per gap bucket — Even / Slight edge / Clear edge / Mismatch — so you can
  see whether controlling for the FIFA-ranking gap sharpens the distribution), or **by
  tournament** (All-time only — two heatmaps, one row per tournament).
- **Scoreline** — show results **order-independent** (2–1 counts a 1–2 too) or oriented
  **favourite → underdog** (using the ranking), which exposes whether the favourite or
  the underdog won.

It shows:

- **Summary cards** — games played, total goals, goals per game, highest-scoring match.
- **Goals per game** — distribution of total goals, splittable by ranking gap.
- **Distinct results** — scoreline frequency (order-independent or favourite-oriented),
  splittable by ranking gap.
- **Rankings** — the higher-ranked team's win rate by ranking gap (with an aggregate
  "All" bar and average goal margin), and the upset rate per tournament.
- **Finished matches** — recent results list.

The page is a single self-contained `index.html` with hand-drawn SVG charts (no
chart library).

## How the data works

Live football APIs can't be called straight from a static page (CORS), and
fifa.com is bot-protected, so instead a small **scraper runs server-side** and
stores results in a flat file that the page reads from the same origin:

```
GitHub Action (every ~15 min, or on demand)
  → scrape.py  (pulls results from ESPN's public JSON feed, one file per edition)
  → commits data/<edition>.json   ← flat files in this repo
  → deploys to GitHub Pages
        → your phone reads data/<edition>.json (same origin, no CORS)
```

- **Editions:** one flat file per tournament — `data/matches.json` (2026),
  `data/2022.json`, `data/2018.json`. The dashboard's selector switches between them.
- **Source:** ESPN's public scoreboard JSON (`site.api.espn.com`, no API key).
  It carries the official World Cup results and is stable to parse. The source is
  isolated in `scrape.py` — swap it there without touching the dashboard.
- **Freshness:** the `Update World Cup data` workflow runs every ~15 minutes. It
  always refreshes the **current** edition (2026) and merges incrementally so a
  file can never blank out; **finished** editions (2022, 2018) are scraped once and
  then skipped (re-scrape with `WC_FORCE=1`).
- **Refresh right now:** repo → **Actions → Update World Cup data → Run workflow**.

## Files

| Path                       | What it is                                              |
| -------------------------- | ------------------------------------------------------ |
| `index.html`               | The dashboard (reads `data/<edition>.json`).           |
| `scrape.py`                | Scraper → writes one file per edition. Stdlib only.    |
| `data/matches.json`        | 2026 results, refreshed by the Action.                 |
| `data/2022.json`, `data/2018.json` | Past tournaments (scraped once).               |
| `../.github/workflows/scrape.yml` | Schedules the scrape + deploys Pages.           |
| `../.github/workflows/pages.yml`  | Deploys Pages on code changes.                  |

## Setup / hosting

1. Merge to the default branch (`main`) — scheduled workflows only run from there.
2. Repo → **Settings → Pages → Source → GitHub Actions** (one-time).
3. Repo → **Actions → Update World Cup data → Run workflow** to populate data immediately
   (otherwise it fills on the next 15-min tick).
4. Open the Pages URL on your phone → **Add to Home Screen**.

### Configuration

Editions (league slug, date window, output file) are defined in the `EDITIONS`
dict at the top of `scrape.py` — add a row there to track another tournament.
Set `WC_FORCE=1` to re-scrape finished editions instead of skipping them.

## Run locally

```bash
cd fifa-world-cup-dashboard
python3 scrape.py            # writes data/matches.json (+ past editions if missing)
python3 -m http.server 8000  # then open http://localhost:8000
```
