# FIFA World Cup Dashboard ⚽

A mobile-first dashboard showing the **accumulated results of the FIFA World Cup**:
goals-per-game distribution, distinct-scoreline frequency, summary stats, and a
results list. Built to open on your phone.

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
  → scrape.py  (pulls results from ESPN's public JSON feed)
  → commits data/matches.json   ← flat file in this repo
  → deploys to GitHub Pages
        → your phone reads data/matches.json (same origin, no CORS)
```

- **Source:** ESPN's public scoreboard JSON (`site.api.espn.com`, no API key).
  It carries the official World Cup results and is stable to parse. The source is
  isolated in `scrape.py` — swap it there without touching the dashboard.
- **Freshness:** the `Update World Cup data` workflow runs every ~15 minutes.
  The phone's **↻ Refresh** button re-reads the flat file so you always see the
  latest committed data.
- **Refresh right now:** repo → **Actions → Update World Cup data → Run workflow**.

## Files

| Path                       | What it is                                              |
| -------------------------- | ------------------------------------------------------ |
| `index.html`               | The dashboard (reads `data/matches.json`).             |
| `scrape.py`                | Scraper → writes `data/matches.json`. Stdlib only.     |
| `data/matches.json`        | Flat data file, updated by the Action.                 |
| `../.github/workflows/scrape.yml` | Schedules the scrape + deploys Pages.           |
| `../.github/workflows/pages.yml`  | Deploys Pages on code changes.                  |

## Setup / hosting

1. Merge to the default branch (`main`) — scheduled workflows only run from there.
2. Repo → **Settings → Pages → Source → GitHub Actions** (one-time).
3. Repo → **Actions → Update World Cup data → Run workflow** to populate data immediately
   (otherwise it fills on the next 15-min tick).
4. Open the Pages URL on your phone → **Add to Home Screen**.

### Configuration

`scrape.py` reads optional env vars (set them in `scrape.yml` if needed):

| Var         | Default      | Meaning                          |
| ----------- | ------------ | -------------------------------- |
| `WC_LEAGUE` | `fifa.world` | ESPN soccer league slug          |
| `WC_START`  | `2026-06-11` | First tournament day (inclusive) |
| `WC_END`    | `2026-07-19` | Last tournament day (inclusive)  |

## Run locally

```bash
cd fifa-world-cup-dashboard
python3 scrape.py            # writes data/matches.json
python3 -m http.server 8000  # then open http://localhost:8000
```
