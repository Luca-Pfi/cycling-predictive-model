"""Previews of upcoming stages: computed as soon as a race is known, refreshed as
the profile, startlist and GC change. Every version is stored, so the track
record can compare what was said a week before, the day before and live.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cyclepredict import config, data


def load_predictor():
    return joblib.load(config.MODELS / "predictor.joblib")


def safe(stage_id):
    return stage_id.replace("/", "__")


def upcoming(stages, today, horizon_days=60):
    s = stages[~stages["finished"].fillna(False).astype(bool)]
    return s[(s["date"] >= today) & (s["date"] <= today + pd.Timedelta(days=horizon_days))]


def preview(predictor, rows, stages, stage_ids):
    ids = list(stage_ids)
    r, s = predictor.predict(rows[rows["stage_id"].isin(ids)], stages[stages["stage_id"].isin(ids)])
    r["finale_rank"] = r.groupby("stage_id")["p_win_if_break"].rank(ascending=False, method="first")
    r["fair_odds_win"] = 1 / r["p_win"].clip(lower=1e-4)
    r["fair_odds_break"] = 1 / r["p_break"].clip(lower=1e-4)
    s["fair_odds_break_wins"] = 1 / s["p_break_wins"].clip(lower=1e-4)
    s["fair_odds_break_caught"] = 1 / (1 - s["p_break_wins"]).clip(lower=1e-4)
    return r, s


def backfill(pred_r, pred_s, stages, folder=config.PREDICTIONS):
    """Store walk-forward predictions as the previews they would have been (the day before)."""
    dates = stages.set_index("stage_id")["date"]
    n = 0
    for sid, g in pred_r.groupby("stage_id"):
        if not any((Path(folder) / safe(sid)).glob("*.json")):
            when = (dates[sid] - pd.Timedelta(days=1)).isoformat()
            n += store(g, pred_s[pred_s["stage_id"] == sid], when=when, folder=folder, backtest=True)
    return n


def store(r, s, when=None, folder=config.PREDICTIONS, backtest=False):
    """Save a version per stage unless nothing changed since the last one."""
    when = when or pd.Timestamp.now().isoformat(timespec="seconds")
    saved = 0
    for sid, g in r.groupby("stage_id"):
        d = Path(folder) / safe(sid)
        d.mkdir(parents=True, exist_ok=True)
        st = s[s["stage_id"] == sid].iloc[0].to_dict()
        payload = g[["rider_id", "team", "p_break", "p_win", "p_win_if_break", "p_win_if_caught"]].round(5)
        digest = hashlib.md5((payload.to_csv(index=False) + f"{st['p_break_wins']:.4f}").encode()).hexdigest()
        versions = sorted(d.glob("*.json"))
        if versions and json.loads(versions[-1].read_text()).get("digest") == digest:
            continue
        stamp = when.replace(":", "-")
        payload.to_csv(d / f"{stamp}.csv", index=False)
        (d / f"{stamp}.json").write_text(json.dumps({"stage_id": sid, "made_at": when, "digest": digest,
                                                     "backtest": backtest,
                                                     "n_starters": int(len(g)),
                                                     **{k: float(v) for k, v in st.items() if k != "stage_id"}}))
        saved += 1
    return saved


def versions(stage_id, folder=config.PREDICTIONS):
    d = Path(folder) / safe(stage_id)
    out = []
    for j in sorted(d.glob("*.json")):
        meta = json.loads(j.read_text())
        riders = pd.read_csv(j.with_suffix(".csv"))
        out.append((meta, riders))
    return out


def refresh(log=print, horizon_days=60):
    """Recompute and store previews for every upcoming stage in the horizon."""
    from cyclepredict.models import prepare
    t, rows, stages = prepare()
    today = data.today()
    up = upcoming(stages, today, horizon_days)
    up = up[up["stage_id"].isin(rows["stage_id"])]  # needs a startlist
    if up.empty:
        log("no upcoming stage with a startlist in the horizon")
        return 0
    r, s = preview(load_predictor(), rows, stages, up["stage_id"])
    when = (today.isoformat() if config.is_demo() else None)
    n = store(r, s, when=when)
    log(f"previews: {len(up)} upcoming stages, {n} new versions stored")
    return n
