"""Track record: score every stored preview once its stage has been raced.

For each finished stage, every stored version is scored (so we see whether the
preview got better as the startlist and GC became known). The last version
before the start is the official pre-race call. Results go to
data/track_record.csv, with notes on the biggest misses.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cyclepredict import config, data, predict

TRACK = config.DATA / "track_record.csv"


def score_version(meta, riders, stage, res):
    r = riders.merge(res[["rider_id", "rank", "in_break"]], on="rider_id", how="left")
    r["in_break"] = r["in_break"].fillna(False).astype(bool)
    won = r[r["rank"] == 1]
    p_bw = meta["p_break_wins"]
    out = dict(stage_id=stage["stage_id"], slug=stage["slug"], date=stage["date"].date().isoformat(),
               made_at=meta["made_at"], p_break_wins=p_bw, labelled=bool(stage["labelled"]))
    if stage["labelled"]:
        y = int(bool(stage["break_won"]))
        out.update(break_won=y, bw_brier=(p_bw - y) ** 2, bw_logloss=-np.log(max(p_bw if y else 1 - p_bw, 1e-4)),
                   break_size=int(stage["break_size"]), expected_break_size=meta.get("expected_break_size"))
        k = int(r["in_break"].sum())
        if k:
            out["break_hits_at_k"] = float(r.nlargest(k, "p_break")["in_break"].mean())
            out["break_logloss"] = float(-np.mean(np.log(np.where(r["in_break"], r["p_break"], 1 - r["p_break"]).clip(1e-4))))
    if len(won):
        w = won.iloc[0]
        out.update(winner=w["rider_id"], p_winner=float(w["p_win"]),
                   winner_rank=int((r["p_win"] > w["p_win"]).sum() + 1),
                   favourite=r.nlargest(1, "p_win")["rider_id"].iloc[0])
    elif stage["finished"]:
        out["note"] = "winner was not on the predicted startlist"
    notes = []
    if stage["labelled"] and abs(p_bw - int(bool(stage["break_won"]))) > 0.6:
        notes.append(f"break {'won' if stage['break_won'] else 'was caught'} against a {100 * p_bw:.0f}% call")
    if len(won) and out["winner_rank"] > 20:
        notes.append(f"winner ranked {out['winner_rank']} in the preview")
    if notes:
        out["note"] = "; ".join(notes)
    return out


def evaluate(log=print):
    """Score all stored previews of finished stages; returns the track record."""
    t, _ = data.load()
    t = data.label_breaks(t)
    st = t["stages"].set_index("stage_id")
    res = t["results"]
    rows = []
    for d in sorted(Path(config.PREDICTIONS).glob("*")):
        sid = d.name.replace("__", "/")
        if sid not in st.index or not bool(st.at[sid, "finished"]):
            continue
        stage = st.loc[sid].copy()
        stage["stage_id"] = sid
        r_stage = res[res["stage_id"] == sid]
        vers = predict.versions(sid)
        start = stage["date"] + pd.Timedelta(hours=11)  # most road stages start late morning
        pre = [i for i, (m, _) in enumerate(vers) if pd.Timestamp(m["made_at"]) <= start]
        for i, (meta, riders) in enumerate(vers):
            row = score_version(meta, riders, stage, r_stage)
            row["version"] = i + 1
            row["backtest"] = bool(meta.get("backtest", False))
            row["official"] = bool(pre) and i == pre[-1]
            row["hours_before"] = round((start - pd.Timestamp(meta["made_at"])).total_seconds() / 3600, 1)
            rows.append(row)
    track = pd.DataFrame(rows)
    if len(track):
        track.to_csv(TRACK, index=False)
        off = track[track["official"]]
        log(f"track record: {off['stage_id'].nunique()} stages scored "
            f"(break-win Brier {off['bw_brier'].mean():.3f}, winner in top 3: "
            f"{(off['winner_rank'] <= 3).mean():.0%})")
    else:
        log("track record: no finished stage with a stored preview yet")
    return track


def summary(track):
    """Accuracy per race, to see where the model is weak."""
    off = track[track["official"]] if "official" in track else track
    if off.empty:
        return off
    return off.groupby("slug").agg(stages=("stage_id", "nunique"), bw_brier=("bw_brier", "mean"),
                                   break_hits=("break_hits_at_k", "mean"),
                                   winner_top3=("winner_rank", lambda x: (x <= 3).mean()),
                                   p_winner=("p_winner", "mean")).reset_index()
