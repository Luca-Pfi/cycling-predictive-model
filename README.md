# Breakaway

Pre-race and live predictions for the road races on the French betting regulator's list
(ANJ decision 2026-159): the WorldTour and other pro races, the Tour de France Femmes, and the
World, European and Olympic road races and time trials. National championships are left out.
The 103 race entries are in [`data/calendar.csv`](data/calendar.csv).

For every stage or one-day race the site shows:

- **Before the race**: will the breakaway win, how big it will be, who will be in it, who wins
  and how (from the break or from the bunch), with fair odds (1 ÷ probability). A preview is
  published as soon as the race is on PCS. Until the startlist is out, last year's riders who are
  still racing stand in for the field (an *early preview*), and every new version is kept.
- **Key paths to victory** for the favourites and the break candidates: how this race is usually
  won, where winning attacks go, the gap a break needs at 50/30/20/10 km, what each rider should do.
- **Live**: pick the riders in the break, the gap, the distance left and the chase; the page gives
  the break's odds, live odds for every rider and live key paths.
- **After the race**: the prediction next to the result, and the accuracy page (season-by-season
  tests, calibration, race by race, biggest misses).

## The website (GitHub Pages)

`.github/workflows/site.yml` runs on every push to `main`, every day, and on demand:
train → store previews of upcoming races → score previews of finished races → commit the previews
back → build the static site → deploy to GitHub Pages.

One-time setting: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

Build it locally:

```bash
pip install -r requirements.txt
python -m cyclepredict site          # writes _site/
python -m http.server -d _site       # http://localhost:8000
```

The site is plain HTML, CSS and JavaScript in [`site/`](site), reading JSON written by
`cyclepredict/site.py`.

## Data

Until real data is added, the site runs on **demo data**: the real calendar with fictional riders
and a fixed demo date (12 July 2026), in exactly the PCS format. It says so at the top of every page.

Real data comes from ProCyclingStats through the [`procyclingstats`](https://github.com/themm1/procyclingstats)
package. PCS blocks automated requests from servers, so fetch on your own computer and push:

```bash
python -m cyclepredict check                     # 1 request: does PCS answer this computer?
python -m cyclepredict verify-calendar           # checks the PCS slug of every race
python -m cyclepredict fetch --years 2019-2025   # history: ~4,000 pages, ~3 h, resumable
git add data/real && git commit -m "PCS data" && git push   # the site rebuilds with real riders
```

During the season, `python -m cyclepredict update` fetches new results, startlists and profiles
for races that are running or within 45 days, then retrains and refreshes the previews. If PCS
refuses your connection, save the pages from your browser and run
`python -m cyclepredict import-pages FOLDER`. With an official PCS API, the fetch step can move
into the GitHub workflow (key stored as a repository secret).

## How it works

| Piece | Details |
|---|---|
| Breakaway labels | PCS records each rider's km ahead of the peloton. In the break: ≥ 20 km and ≥ 25% of the distance. The break *won* if the winner was in it. Races without this data only train the winner models. |
| Features ([`features.py`](cyclepredict/features.py)) | All computed from races *before* the one predicted (one as-of join for training and previews). Rider: break habits by terrain and over the last year, results on similar terrain, form, record in this race, team habits, GC position before the stage, KOM activity, yesterday's break, role in team, home race, age. Stage: profile, climbs, position in the race, field strength, GC gaps. Race: its own break history, shrunk toward the terrain average. |
| Models ([`models.py`](cyclepredict/models.py)) | Gradient boosting: break wins (calibrated), break size, in the break, wins if the break survives, wins if it is caught. P(win) = P(break wins) × P(in break) × P(wins the break) + P(caught) × P(wins from the bunch). Tested season by season on races the models had not seen. |
| Live ([`live.py`](cyclepredict/live.py)) | Chase model: the peloton closes about 6 s/km with several teams chasing, more in the final 10 km, less against a big break; blended with the pre-race call early in the race. Refitted on logged live estimates once 30 races are logged. |
| Key paths ([`insights.py`](cyclepredict/insights.py)) | Race history (sprint / small group / solo, solo distances, winning break size), rider type from results by terrain, and the gap-needed curve. |
| Track record ([`evaluate.py`](cyclepredict/evaluate.py)) | Every stored preview is scored after its race; misses are flagged. Previews live in `data/predictions/` and are committed. |

## Known limits

- Live input is entered by hand: the package cannot read the PCS live ticker. A saved PCS live
  page would let an automatic feed be built.
- The live model uses rule-of-thumb coefficients until enough live estimates are logged.
- PCS slugs marked `unverified` in the calendar must be checked with `verify-calendar`.
- PCS breakaway km may be missing for older seasons.
- Fair odds have no bookmaker margin, and probabilities are model estimates: check the accuracy page.
