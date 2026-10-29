"""Shared schema + leak-free feature engineering.

Data schema (same for the synthetic sample and the real PCS scrape):
  stages.csv  : stage_id, race, year, date, distance_km, elevation_gain_m,
                profile_score, uphill_finish, is_itt, stage_type
  riders.csv  : rider_id, name, team, birth_year
  results.csv : stage_id, rider_id, rank
"""
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_FEATURES = ["distance_km", "elevation_gain_m", "profile_score", "uphill_finish", "is_itt"]
STAGE_TYPES = ["flat", "hilly", "mountain", "itt"]

RIDER_FEATURES = [
    "sim_pct",       # similarity-weighted mean percentile on similar past stages
    "sim_top10",     # similarity-weighted top-10 rate on similar past stages
    "sim_weight",    # how much experience the rider has on similar stages
    "sim_best",      # best percentile on a clearly similar stage
    "recent_pct",    # mean percentile over the last 15 starts (any terrain)
    "n_starts",
    "age",
]
MODEL_FEATURES = STAGE_FEATURES + RIDER_FEATURES

# Scale used to standardise stage vectors before measuring stage similarity.
_SCALE = np.array([60.0, 1200.0, 80.0, 0.7, 0.35])
_BANDWIDTH = 1.0
_SHRINK = 2.0  # pseudo-observations pulling sparse histories towards average
RECENT_N = 15


def load_data(data_dir):
    d = Path(data_dir)
    stages = pd.read_csv(d / "stages.csv", parse_dates=["date"])
    riders = pd.read_csv(d / "riders.csv")
    results = pd.read_csv(d / "results.csv")
    stages = stages.sort_values(["date", "stage_id"]).reset_index(drop=True)
    return stages, riders, results


def stage_matrix(stages):
    return stages[STAGE_FEATURES].to_numpy(dtype=float) / _SCALE


def similarity(stage_vec, hist_matrix):
    """Gaussian-kernel similarity between one stage and many past stages."""
    d2 = (((hist_matrix - stage_vec / _SCALE) ** 2)).sum(axis=1)
    return np.exp(-d2 / (2 * _BANDWIDTH**2))


def rider_features(sim_row, hist_idx, hist_pct, hist_top10, age):
    """Features for one rider given their history (indices into sim_row)."""
    n = len(hist_idx)
    if n == 0:
        return [0.5, 0.08, 0.0, 0.0, 0.5, 0, age]
    w = sim_row[hist_idx]
    sw = w.sum()
    close = w > 0.5
    return [
        (w @ hist_pct + _SHRINK * 0.5) / (sw + _SHRINK),
        (w @ hist_top10 + _SHRINK * 0.08) / (sw + _SHRINK),
        sw,
        hist_pct[close].max() if close.any() else 0.0,
        hist_pct[-RECENT_N:].mean(),
        n,
        age,
    ]


def add_percentiles(results):
    n = results.groupby("stage_id")["rank"].transform("max")
    results = results.copy()
    results["pct"] = 1 - (results["rank"] - 1) / (n - 1).clip(lower=1)
    results["top10"] = (results["rank"] <= 10).astype(int)
    return results


def build_training_table(stages, riders, results):
    """One row per (stage, starter) using only results from earlier dates."""
    results = add_percentiles(results)
    stage_pos = {sid: i for i, sid in enumerate(stages["stage_id"])}
    X = stage_matrix(stages)
    birth = riders.set_index("rider_id")["birth_year"].to_dict()
    by_stage = {sid: g for sid, g in results.groupby("stage_id")}

    history = {}  # rider_id -> ([stage_pos], [pct], [top10])
    rows = []
    for _, day in stages.groupby("date", sort=True):
        pending = []
        for _, st in day.iterrows():
            g = by_stage.get(st.stage_id)
            if g is None:
                continue
            i = stage_pos[st.stage_id]
            sim_row = similarity(X[i] * _SCALE, X)
            for r in g.itertuples():
                h = history.get(r.rider_id, ([], [], []))
                feats = rider_features(
                    sim_row,
                    np.asarray(h[0], dtype=int),
                    np.asarray(h[1]),
                    np.asarray(h[2]),
                    st.year - birth.get(r.rider_id, st.year - 27),
                )
                rows.append([st.stage_id, r.rider_id, st.date, r.rank, r.top10,
                             *st[STAGE_FEATURES].tolist(), *feats])
                pending.append((r.rider_id, i, r.pct, r.top10))
        # results of this day only become history after the whole day is done
        for rid, i, pct, t10 in pending:
            h = history.setdefault(rid, ([], [], []))
            h[0].append(i)
            h[1].append(pct)
            h[2].append(t10)

    cols = ["stage_id", "rider_id", "date", "rank", "top10"] + MODEL_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    df[STAGE_FEATURES] = df[STAGE_FEATURES].astype(float)
    return df


def features_for_new_stage(stage, stages, riders, results, rider_ids, season):
    """Rider features for a hypothetical stage using the full history."""
    results = add_percentiles(results)
    stage_pos = {sid: i for i, sid in enumerate(stages["stage_id"])}
    results = results[results["stage_id"].isin(stage_pos)].copy()
    results["pos"] = results["stage_id"].map(stage_pos)
    results = results.sort_values("pos")
    sim_row = similarity(np.array([stage[f] for f in STAGE_FEATURES], dtype=float),
                         stage_matrix(stages))
    birth = riders.set_index("rider_id")["birth_year"].to_dict()
    by_rider = {rid: g for rid, g in results.groupby("rider_id")}

    rows = []
    for rid in rider_ids:
        g = by_rider.get(rid)
        if g is None:
            continue
        feats = rider_features(sim_row, g["pos"].to_numpy(), g["pct"].to_numpy(),
                               g["top10"].to_numpy(), season - birth[rid])
        rows.append([rid, *[stage[f] for f in STAGE_FEATURES], *feats])
    df = pd.DataFrame(rows, columns=["rider_id"] + MODEL_FEATURES)
    df[STAGE_FEATURES] = df[STAGE_FEATURES].astype(float)
    return df, sim_row
