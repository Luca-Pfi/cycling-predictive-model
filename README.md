# 🚴 Who Suits This Stage?

Design a pro-cycling stage (distance, climbing, hardness, uphill finish, time trial)
and a machine-learning model predicts the stage type and which riders are most
likely to finish in the top 10.

## Quick start

```bash
pip install -r requirements.txt
python -m src.make_sample_data      # synthetic dataset -> data/sample
python -m src.train                 # train + evaluate  -> models/
streamlit run app.py
```

The repo already contains the sample data and trained models, so `streamlit run app.py`
works right after `pip install`. If you see a scikit-learn version warning, retrain with
`python -m src.train`.

## Using real ProCyclingStats data

```bash
python -m src.scrape_pcs --years 2019-2024 --out data/pcs   # slow + polite, results cached
python -m src.train --data data/pcs
streamlit run app.py
```

The app reads whichever dataset the last `train` run used.

## How it works

| File | What it does |
|---|---|
| `src/make_sample_data.py` | Fictional riders with hidden abilities (sprint, punch, climb, TT, age curve, form) and races with realistic stage profiles. |
| `src/scrape_pcs.py` | Real stage-race results from PCS in the same schema. |
| `src/features.py` | Leak-free features: for each stage, every rider's record on **similar past stages** (Gaussian-kernel similarity on stage profiles), recent form, experience and age. Only results from earlier dates are used. |
| `src/train.py` | **Model A**: Random Forest, stage profile → type. **Model B**: gradient boosting, stage + rider features → P(top 10). Held-out last season, with a "current form only" baseline. |
| `app.py` | Streamlit app: stage designer, model performance (metrics, feature importance, calibration) and a rider explorer. |

## Ideas to extend

- Add features: gradient of the final km, climbs in the last 20 km, cobbles, weather.
- Team effects: a strong lead-out train for sprinters, or team strength for GC riders.
- Try LightGBM `LGBMRanker` (learning to rank) instead of a classifier.
- Use real startlists (`procyclingstats.RaceStartlist`) to predict upcoming races.
