# 🚴 Pro Cycling Predictor

A two-page Streamlit app with machine-learning models for pro cycling:

1. **🏔️ Who suits this stage?** Design a stage (distance, climbing, hardness, uphill finish,
   time trial). The app predicts the stage type and which riders are most likely to finish in the top 10.
2. **📈 Career trajectory & talent scouting.** A penalised **linear regression** with
   ~45 variables projects a young rider's PCS points at every age up to 38, with prediction
   intervals, the probability they are still racing, a "future stars" leaderboard, a what-if
   simulator and full model diagnostics.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repo already contains the sample datasets and trained models. To rebuild everything
(for example after a scikit-learn version warning):

```bash
python -m src.make_sample_data      # stage dataset   -> data/sample
python -m src.train                 # stage models    -> models/
python -m src.make_career_data      # career dataset  -> data/careers
python -m src.train_career          # career models   -> models/career   (~20 min)
```

Both datasets are **synthetic** (fictional riders and races) but built to behave like the real
sport. Scrapers for real ProCyclingStats data are included (see below).

## Page 2: the career model in detail

**Question.** Given everything known about a rider at a *cutoff age* (21–25), how many PCS
points will they score at each later age?

**Variables (~45 raw):** VO2max, FTP at 19 (W/kg), 5-s sprint power (W and W/kg), height,
weight, BMI, age started cycling, background (road/MTB/cyclocross/track/triathlon),
nationality, specialty, Junior Worlds top 10, Junior Nations Cup points, development team,
U23 wins and podiums, best Tour de l'Avenir and U23 Worlds rank, points in the cutoff season
and the two before, mean / best / trend of log points, career points, wins, top-10s,
top-10s per race day, race days per season, Grand Tour starts, WorldTeam / ProTeam seasons,
team-level changes, current team level, injury seasons, injured in cutoff season, and
missing-value flags.

**Model design (`src/career_model.py`).**
- A long panel with one row per *rider × future age*; target = `log(1 + points)`.
- Median imputation + missing indicators, standardisation, one-hot encoding.
- **Cubic spline of the target age** (the average career curve).
- **Spline × variable interactions**, so a variable can change the *shape* of the curve:
  late starters peak later, sprinters peak earlier, strong U23 riders rise faster.
  That makes **~160 columns**, all still a linear model.
- **OLS, Ridge, Lasso and Elastic Net** compared with 5-fold **rider-grouped** CV
  (a rider is never in both training and validation), plus a baseline that uses only
  age + current points.
- **Held-out test set:** 20% of riders, whose whole careers are never seen in training.
- **Survivorship:** a logistic regression for P(still racing at age *a*). Weak riders
  retire, so their missing seasons are not random.
- **Prediction intervals:** split-conformal, using out-of-fold residual quantiles at each
  horizon (1–17 years ahead).
- **Inference:** a separate OLS on standardised main effects with **rider-clustered
  standard errors**, giving 95% CIs and p-values. Variables with **VIF > 10** are removed
  first (e.g. the points trend is an exact combination of the last three seasons).

**Results at cutoff age 23 (held-out riders):** see the *Model insight* tab. Typical
numbers on the sample data: test R² ≈ 0.76 (baseline 0.62), 80% intervals cover ≈ 80% of
outcomes, survival AUC ≈ 0.88, and rank correlation of predicted vs real career peak ≈ 0.85.

**App tabs:** 🌟 future stars leaderboard · 📈 rider projection (with the 5 most similar
historical riders, and honest backtests on held-out riders) · 🎛️ what-if simulator ·
🔬 model insight (model comparison, coefficient plot with CIs, VIF, Lasso terms, residual
diagnostics, Q-Q plot, accuracy vs cutoff age).

## Using real ProCyclingStats data

Both scrapers use the [`procyclingstats`](https://github.com/themm1/procyclingstats) package and
keep every page as raw HTML in a cache (`data/pcs*/cache`), so a page is fetched at most once and
runs can be stopped and resumed.

**Option A: let the package download the pages** (polite, 1.5 s between requests):

```bash
python -m src.scrape_pcs_careers --years 2010-2025 --top 300   # -> data/pcs_careers
python -m src.train_career --data data/pcs_careers
python -m src.scrape_pcs --years 2019-2025                      # -> data/pcs
python -m src.train --data data/pcs
streamlit run app.py                                            # now shows real riders
```

PCS is behind Cloudflare. If the package reports "Cloudflare protection detected", PCS is
refusing automated requests from your connection; use option B instead of trying to get around it.

**Option B: pages you saved in your browser.** Open the PCS pages you want (rider pages for the
career model; race overview and stage pages for the stage model), use *Save page as...*
(HTML only) into one folder, then:

```bash
python -m src.scrape_pcs_careers --import-dir ~/Downloads/pcs --offline
python -m src.scrape_pcs --import-dir ~/Downloads/pcs --offline
```

Each saved file is recognised by its canonical PCS link, so file names don't matter.
The parsers are tested against real PCS pages from the package's test fixtures.

PCS does not publish lab tests, junior or U23 results, so on real data the career model uses the
variables PCS has (age, nationality, height, weight, points per season, team level). On
PCS-shaped data the test R² is about 0.63 (vs 0.54 for the baseline), compared with 0.77 when the
extra variables are available. Please respect the PCS terms of use.

## Project layout

| File | What it does |
|---|---|
| `app.py` | Streamlit entry point (navigation between the two pages) |
| `views/stage_designer.py` | Page 1 |
| `views/career.py` | Page 2 |
| `src/make_sample_data.py`, `src/features.py`, `src/train.py` | Stage data, similarity features, Random Forest + gradient boosting |
| `src/make_career_data.py` | Synthetic careers: hidden talent, early/late bloomers, team feedback loops, injuries, retirement |
| `src/career_model.py` | Career features, design matrix (splines + interactions), projection |
| `src/train_career.py` | Model comparison, inference, VIF, conformal intervals, survival model |
| `src/scrape_pcs.py`, `src/scrape_pcs_careers.py`, `src/pcs_pages.py` | Real data from ProCyclingStats (download or saved pages, HTML cache) |
