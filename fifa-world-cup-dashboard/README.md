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
have different numbers of matches. Both charts also carry a cumulative **Pareto** line.

In the **All-time** view a **Display** toggle switches between **Aggregate** (the
combined bar charts) and **Compare** — two heatmaps (goals-per-game and scorelines)
with one row per tournament, each cell shaded by that tournament's share of matches,
so you can compare 2018 / 2022 / 2026 side by side. The heatmaps respect the stage
and completion filters too.

It shows:

- **Summary cards** — games played, total goals, goals per game, highest-scoring match.
- **Goals per game** — distribution of how many matches finished with each total goal count.
- **Distinct results** — frequency of each scoreline (order-independent, so 2–1 also counts 1–2).
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
