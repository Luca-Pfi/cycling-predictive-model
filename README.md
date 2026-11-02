# 🚴 Breakaway Predictor

Pre-race and live predictions for the road races on the French betting regulator's list
(ANJ decision 2026-159): the WorldTour, ProSeries and other pro races, the Tour de France Femmes,
and the Worlds / Europeans / Olympics (road race and time trial, men and women). National
championships are left out. The 108 races are in [`data/calendar.csv`](data/calendar.csv).

For every stage or one-day race:

- **Pre-race** (as soon as PCS publishes the race, refreshed as the profile, startlist and GC change):
  will the breakaway win (with fair odds), how big will it be, who will be in it, who wins, and by
  which route (from the break or from the bunch). Every rider gets fair decimal odds = 1 / probability.
- **Key paths to victory** for the favourites and the break candidates: how this race is usually
  won, where winning attacks go, the gap a break needs at 50/30/20/10 km, and what each rider type
  should do.
- **Live**: enter the break riders, gap, km to go and chase. You get the break's survival odds,
  live odds for every rider, the "gap needed" curve and live key paths.
- **Track record**: every preview is stored and scored once the race is over (per race, per stage,
  calibration, biggest misses). The models are then retrained, so each new race improves the next.

## Quick start (demo data, works offline)

```bash
pip install -r requirements.txt
python -m cyclepredict demo        # synthetic seasons in the exact PCS format (fictional riders)
python -m cyclepredict train       # walk-forward evaluation + final models (~1 min)
python -m cyclepredict preview     # previews of upcoming stages
python -m cyclepredict evaluate    # track record
streamlit run app.py
```

The demo uses the real calendar with **fictional riders** and a fixed demo clock (12 July 2026),
so every screen can be tried before real data arrives. The app shows a clear DEMO banner.

## Real data: ProCyclingStats via the `procyclingstats` package

PCS is behind Cloudflare. The package downloads pages normally, one every 2 s, and caches them as
HTML in `data/pcs_pages/`. If PCS refuses your connection, don't try to get around it: save the
pages from your browser and run `import-pages` instead.

```bash
python -m cyclepredict check                     # 1 request: does PCS answer this computer?
python -m cyclepredict verify-calendar           # checks the PCS slug of every race
python -m cyclepredict fetch --years 2019-2025   # history: ~4,000 pages, ~3 h, resumable
python -m cyclepredict train
python -m cyclepredict update                    # each race day (see below)
streamlit run app.py                             # switches to real data automatically
```

`update` does the whole loop: fetch new results, startlists and profiles for races that are running
or within 45 days → score the stored previews against the results → retrain → store fresh previews.

## How it works

| Piece | Details |
|---|---|
| Breakaway labels | PCS records each rider's `breakaway_kms`. A rider was in the break with ≥ max(20 km, 25% of the distance); the break *won* if the winner was in it. Races without breakaway km are used for the winner models only. |
| Features (`cyclepredict/features.py`) | Built "as of" the race date with one as-of join, so training rows and previews use the same code and never see the future. **Rider**: break rate (overall, by terrain, last 12 months), results by terrain, form, record in this race, team break habits, GC rank/gap before the stage, KOM activity, in yesterday's break, role in team, home race, age, points. **Stage**: profile, climbs, stage position, startlist quality, GC tightness, sprinters/climbers/attackers in the field. **Race priors**: this race's own break-win history, shrunk toward the terrain average, so every race gets its own baseline. |
| Models (`cyclepredict/models.py`) | Gradient boosting: P(break wins) (isotonic-calibrated), break size, P(in break), P(win \| break survives, in it), P(win \| break caught). Combined: P(win) = P(bw)·P(in break)·P(win in break) + (1−P(bw))·P(win from bunch). |
| Live (`cyclepredict/live.py`) | Chase model: the peloton closes ~6 s/km when several teams ride (more in the last 10 km, less with one team, less against a big break). Survival = gap vs closing over the remaining distance, with uncertainty, blended with the pre-race call early in the race. Saved live estimates are scored after the stage; with ≥ 30 logged races the live model is refit on them. |
| Key paths (`cyclepredict/insights.py`) | Race history (sprint / small group / solo shares, solo distances, winning break size), rider type from results by terrain, and the gap-needed curve. |
| Track record (`cyclepredict/evaluate.py`) | Brier score of the break call, break picks hit at k, winner rank and probability, per race; misses are flagged. The backtest (walk-forward seasons) is stored the same way. |

**Demo results** (walk-forward, 2024–2026). These numbers only test the machinery; they say nothing
yet about real races:

| | Model | Baseline |
|---|---|---|
| Break wins: AUC | 0.77 | 0.72 (race history) |
| Break riders: AUC | 0.87 | 0.84 (rider's break rate) |
| Break picks that were in the break (top k) | 34% | |
| Winner in top 3 | 70% | 70% (form) |

## Known limits

- **Live input is manual.** The package has no parser for PCS's live ticker. Send a saved PCS live
  page and an automatic feed can be added.
- The live chase model uses rule-of-thumb coefficients until enough live snapshots are logged.
- PCS slugs marked `unverified` in the calendar must be checked with `verify-calendar`.
- PCS breakaway km may be missing for older seasons; check the Data tab after fetching.
- Rider pages give career specialty points *as of today*. Those are used only for display, never as
  model features, because they would leak future results into backtests.
- Fair odds have no bookmaker margin, and probabilities are model estimates: check the track record
  before trusting them.

`explore_pcs.py` and `pcs_data_catalog.md` list every field the `procyclingstats` package can extract.
