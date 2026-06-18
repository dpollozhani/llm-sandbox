# FIFA World Cup Dashboard ⚽

A single-file, mobile-first dashboard that shows the **accumulated results of the
current FIFA World Cup**, and updates **on demand** (a Refresh button pulls live
scores). Built to open on your phone.

It shows:

- **Summary cards** — games played, total goals, goals per game, highest-scoring match.
- **Goals per game** — distribution of how many matches finished with each total goal count.
- **Distinct results** — frequency of each scoreline (order-independent, so 2–1 also counts 1–2).
- **Finished matches** — recent results list.

There is no build step and no chart library: it's one self-contained `index.html`
with hand-drawn SVG charts. The only network call is the data fetch, which happens
in **your browser** (so it works on your phone regardless of where the file is hosted).

## Data source

Data comes from [TheSportsDB](https://www.thesportsdb.com/). Defaults:

| Setting    | Default | Meaning                          |
| ---------- | ------- | -------------------------------- |
| League ID  | `4429`  | FIFA World Cup                   |
| Season     | `2026`  | 2026 tournament                  |
| API key    | `3`     | Free public test key            |

Open the **Data settings** panel in the app to change these. Values are saved on
your device. The app tries the full-season endpoint first and falls back to the
league's most recent matches (which works on the free tier). If a season shows no
data, double-check the season string format under settings, or
[grab your own API key](https://www.thesportsdb.com/api.php).

## Put it on your phone (GitHub Pages)

1. Merge this folder to the repository's **default branch** (`main`).
2. In GitHub, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions**.
4. The included workflow (`.github/workflows/pages.yml`) deploys automatically.
   When it finishes, your URL appears in **Settings → Pages** (usually
   `https://<you>.github.io/<repo>/`).
5. Open that URL on your phone and **Add to Home Screen** for an app-like icon.

## Run it locally

Just open `index.html` in any browser, or serve the folder:

```bash
cd fifa-world-cup-dashboard
python3 -m http.server 8000
# then visit http://localhost:8000
```
