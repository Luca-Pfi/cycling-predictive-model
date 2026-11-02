"""Export previews, live-model inputs and the track record as one JSON file,
for the standalone web page (no Python needed to browse it).

    python -m cyclepredict.export out.json
"""
import json
import sys

import numpy as np
import pandas as pd

from cyclepredict import config, data, evaluate, features, insights, live, models, predict


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def build(horizon_days=60):
    t, rows, stages = models.prepare()
    P = predict.load_predictor()
    today = data.today()
    hist = features.rider_history(t)
    prof = insights.rider_profiles(hist, today)
    names = t["riders"].set_index("rider_id")["name"].to_dict()
    cal = pd.read_csv(config.CALENDAR).set_index("pcs_slug")
    up = predict.upcoming(stages, today, horizon_days)
    up = up[up["stage_id"].isin(rows["stage_id"])]
    r_all, s_all = predict.preview(P, rows, stages, up["stage_id"])
    fo = P.finale.predict_proba(rows[models.WIN_FEATURES])[:, 1]
    fav = P.favourite.predict_proba(rows[models.WIN_FEATURES])[:, 1]
    raw = rows[["stage_id", "rider_id"]].assign(fo=fo / (1 - fo + 1e-9), fav=fav)
    r_all = r_all.merge(raw, on=["stage_id", "rider_id"])
    T = t["stages"].set_index("stage_id")
    out_stages = []
    for sid in up.sort_values("date")["stage_id"]:
        S = T.loc[sid]
        stage = {**S.to_dict(), "stage_id": sid}
        r = r_all[r_all["stage_id"] == sid].sort_values("p_win", ascending=False)
        s0 = s_all[s_all["stage_id"] == sid].iloc[0]
        pat = insights.race_patterns(t["stages"], S["slug"], today, features.ICON_GROUP.get(S["profile_icon"]))
        p_bw = float(s0["p_break_wins"])
        top = list(r.head(5)["rider_id"])
        picks = top + [x for x in r.sort_values("p_break", ascending=False)["rider_id"] if x not in top][:5]
        paths = {}
        for rid in picks:
            row = r[r["rider_id"] == rid].iloc[0].to_dict()
            pr = prof.loc[rid].to_dict() if rid in prof.index else None
            paths[rid] = insights.key_paths(stage, row, pr, pat, p_bw)
        climbs = {k.replace("climbs_", "").upper(): int(S[k]) for k in ["climbs_hc", "climbs_1", "climbs_2", "climbs_3", "climbs_4"] if S[k]}
        out_stages.append(dict(
            id=sid, race=cal["name"].get(S["slug"], S["slug"]), year=int(S["year"]),
            stage=int(S["stage_index"]) if S["n_stages"] > 1 else None, n_stages=int(S["n_stages"]),
            date=S["date"].date().isoformat(), km=_r(S["distance_km"], 1), vert=int(S["vertical_m"]),
            ps=int(S["profile_score"]), icon=S["profile_icon"], itt=S["stage_type"] == "ITT", climbs=climbs,
            last_climb=S["last_climb"] if isinstance(S["last_climb"], str) else None,
            last_cat=S["last_climb_cat"] if isinstance(S["last_climb_cat"], str) else None,
            p_bw=_r(p_bw), size=_r(s0["expected_break_size"], 1),
            pattern={k: (_r(v, 3) if not isinstance(v, tuple) else [_r(x, 1) for x in v]) for k, v in pat.items()},
            riders=[dict(id=x.rider_id, n=names.get(x.rider_id, x.rider_id), t=x.team,
                         ty=prof.loc[x.rider_id, "type"] if x.rider_id in prof.index else "rider",
                         pb=_r(x.p_break), pw=_r(x.p_win, 5), qb=_r(x.p_win_if_break, 5), qc=_r(x.p_win_if_caught, 5),
                         fo=_r(x.fo, 6), fv=_r(x.fav, 6), fr=int(x.finale_rank))
                    for x in r.itertuples()],
            paths=paths))
    meta = json.loads((config.MODELS / "metrics.json").read_text())
    track = pd.read_csv(evaluate.TRACK) if evaluate.TRACK.exists() else pd.DataFrame()
    off = track[track["official"]] if len(track) else track
    calib, per_race, misses = [], [], []
    if len(off):
        lab = off.dropna(subset=["break_won"])
        b = pd.cut(lab["p_break_wins"], [0, .05, .1, .2, .3, .4, .5, .6, .8, 1])
        calib = [dict(pred=_r(g["p_break_wins"].mean()), obs=_r(g["break_won"].mean()), n=int(len(g)))
                 for _, g in lab.groupby(b, observed=True)]
        summ = evaluate.summary(track)
        per_race = [dict(race=cal["name"].get(x.slug, x.slug), stages=int(x.stages), brier=_r(x.bw_brier, 3),
                         hits=_r(x.break_hits, 3), top3=_r(x.winner_top3, 3), pw=_r(x.p_winner, 3))
                    for x in summ.itertuples()]
        m = off[off["note"].notna()].sort_values("date", ascending=False).head(12)
        misses = [dict(date=x.date, race=cal["name"].get(x.slug, x.slug), p=_r(x.p_break_wins, 3), note=x.note)
                  for x in m.itertuples()]
    return dict(demo=config.is_demo(), today=today.date().isoformat(), metrics=meta, stages=out_stages,
                calib=calib, per_race=per_race, misses=misses,
                live=dict(chase=live.CHASE, sd=live.LiveModel().sd, margin=live.LiveModel().finale_margin,
                          size_effect=live.LiveModel().size_effect))


if __name__ == "__main__":
    out = build()
    with open(sys.argv[1] if len(sys.argv) > 1 else "web_data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"), default=str)
    print(f"{len(out['stages'])} stages exported")
