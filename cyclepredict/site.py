"""Build the static website (GitHub Pages) from the data, models and stored previews.

    python -m cyclepredict site            # -> _site/
    python -m http.server -d _site         # look at it locally

Static files come from site/ (index.html, app.js, style.css); this module writes
the JSON they read: data/index.json (season calendar), data/race/<id>.json (one
per race: every stage, preview or prediction-vs-result), data/accuracy.json.
"""
import json
import shutil

import numpy as np
import pandas as pd

from cyclepredict import config, data, evaluate, features, insights, live, models, predict

SRC = config.ROOT / "site"


def _r(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, n)


def race_key(race_id):
    return race_id.replace("/", "-")


def _stage_base(S, sid):
    climbs = {k.replace("climbs_", "").upper(): int(S[k]) for k in ["climbs_hc", "climbs_1", "climbs_2", "climbs_3", "climbs_4"]
              if pd.notna(S[k]) and S[k]}
    return dict(id=sid, n=int(S["stage_index"]), date=S["date"].date().isoformat(), km=_r(S["distance_km"], 1),
                vert=None if pd.isna(S["vertical_m"]) else int(S["vertical_m"]),
                ps=None if pd.isna(S["profile_score"]) else int(S["profile_score"]), icon=S["profile_icon"],
                itt=S["stage_type"] == "ITT", climbs=climbs,
                last_climb=S["last_climb"] if isinstance(S["last_climb"], str) else None,
                last_cat=S["last_climb_cat"] if isinstance(S["last_climb_cat"], str) else None)


def _official(sid, start):
    vers = predict.versions(sid)
    pre = [(m, r) for m, r in vers if pd.Timestamp(m["made_at"]) <= start]
    return (pre[-1] if pre else None), vers


def build(out=config.ROOT / "_site", season=None, log=print):
    t, rows, stages = models.prepare()
    P = predict.load_predictor()
    today = data.today()
    season = season or today.year
    hist = features.rider_history(t)
    prof = insights.rider_profiles(hist, today)
    names = t["riders"].set_index("rider_id")["name"].to_dict()
    cal = pd.read_csv(config.CALENDAR).set_index("pcs_slug")
    T = t["stages"].set_index("stage_id")
    res = t["results"]

    season_st = stages[stages["year"] == season]
    open_ids = season_st[~season_st["finished"].fillna(False).astype(bool) & season_st["stage_id"].isin(rows["stage_id"])]["stage_id"]
    r_up, s_up = predict.preview(P, rows, stages, open_ids) if len(open_ids) else (pd.DataFrame(), pd.DataFrame())
    if len(r_up):
        sub = rows[rows["stage_id"].isin(open_ids)]
        fo = P.finale.predict_proba(sub[models.WIN_FEATURES])[:, 1]
        r_up = r_up.merge(sub[["stage_id", "rider_id"]].assign(fo=fo / (1 - fo + 1e-9),
                          fv=P.favourite.predict_proba(sub[models.WIN_FEATURES])[:, 1]), on=["stage_id", "rider_id"])

    from pathlib import Path
    out = Path(out) if str(out).startswith("/") else config.ROOT / out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC, out)
    (out / ".nojekyll").write_text("")
    (out / "data" / "race").mkdir(parents=True, exist_ok=True)

    index = []
    races = t["races"][t["races"]["year"] == season].sort_values("startdate")
    for race in races.itertuples():
        rs = season_st[season_st["race_id"] == race.race_id].sort_values("stage_index")
        if rs.empty:
            continue
        key = race_key(race.race_id)
        riders, out_stages = set(), []
        for sid in rs["stage_id"]:
            S = T.loc[sid]
            st = _stage_base(S, sid)
            start = S["date"] + pd.Timedelta(hours=11)
            if bool(S["finished"]):
                off, vers = _official(sid, start)
                if off is not None:
                    meta, pr = off
                    st["call"] = dict(p_bw=_r(meta["p_break_wins"]), size=_r(meta.get("expected_break_size"), 1),
                                      at=meta["made_at"][:10], backtest=bool(meta.get("backtest")), retro=False)
                elif sid in set(rows["stage_id"]):  # no stored preview: computed after the race, flagged
                    pr, ps = P.predict(rows[rows["stage_id"] == sid], stages[stages["stage_id"] == sid])
                    st["call"] = dict(p_bw=_r(ps["p_break_wins"].iloc[0]), size=_r(ps["expected_break_size"].iloc[0], 1),
                                      at=None, backtest=False, retro=True)
                else:
                    pr = None
                rr = res[res["stage_id"] == sid]
                won = rr[rr["rank"] == 1]
                result = dict(how=S["won_how"] if isinstance(S["won_how"], str) else None,
                              winner=won["rider_id"].iloc[0] if len(won) else None,
                              top10=list(rr[rr["rank"] <= 10].sort_values("rank")["rider_id"]),
                              brk=list(rr[rr["in_break"]]["rider_id"]) if bool(S.get("labelled")) else None,
                              break_won=bool(S.get("break_won")) if bool(S.get("labelled")) else None)
                riders |= set(result["top10"]) | set(result["brk"] or [])
                if pr is not None and len(pr):
                    pr = pr.sort_values("p_win", ascending=False)
                    top = pd.concat([pr.head(25), pr.nlargest(15, "p_break")]).drop_duplicates("rider_id")
                    st["riders"] = [[x.rider_id, _r(x.p_break), _r(x.p_win, 5), _r(x.p_win_if_break, 5), _r(x.p_win_if_caught, 5)]
                                    for x in top.itertuples()]
                    riders |= set(top["rider_id"])
                    if result["winner"]:
                        p = pr.set_index("rider_id")["p_win"]
                        pw = float(p.get(result["winner"], 0))
                        result.update(winner_p=_r(pw, 5), winner_rank=int((p > pw).sum() + 1))
                    if result["brk"]:
                        k = len(result["brk"])
                        result["break_hits"] = int(pr.nlargest(k, "p_break")["rider_id"].isin(result["brk"]).sum())
                st.update(status="finished", result=result)
            else:
                st["status"] = "upcoming"
                g = r_up[r_up["stage_id"] == sid] if len(r_up) else pd.DataFrame()
                if g.empty:
                    st["note"] = "No field known yet."
                else:
                    s0 = s_up[s_up["stage_id"] == sid].iloc[0]
                    g = g.sort_values("p_win", ascending=False)
                    pat = insights.race_patterns(t["stages"], S["slug"], today, features.ICON_GROUP.get(S["profile_icon"]))
                    stage = {**S.to_dict(), "stage_id": sid}
                    p_bw = float(s0["p_break_wins"])
                    favs = list(g.head(3)["rider_id"])
                    picks = favs + [x for x in g.sort_values("p_break", ascending=False)["rider_id"] if x not in favs][:4]
                    paths = {}
                    for rid in picks:
                        row = g[g["rider_id"] == rid].iloc[0].to_dict()
                        paths[rid] = insights.key_paths(stage, row, prof.loc[rid].to_dict() if rid in prof.index else None, pat, p_bw)
                    st.update(provisional=bool(s0["provisional"]), p_bw=_r(p_bw), size=_r(s0["expected_break_size"], 1),
                              pattern={k: (_r(v, 3) if not isinstance(v, tuple) else [_r(x, 1) for x in v]) for k, v in pat.items()},
                              riders=[[x.rider_id, _r(x.p_break), _r(x.p_win, 5), _r(x.p_win_if_break, 5), _r(x.p_win_if_caught, 5),
                                       _r(x.fo, 6), _r(x.fv, 6)] for x in g.itertuples()],
                              favs=favs, paths=paths)
                    riders |= set(g["rider_id"])
                st["versions"] = [dict(at=m["made_at"][:10], p_bw=_r(m["p_break_wins"]), prov=bool(m.get("provisional")),
                                       fav=v.nlargest(1, "p_win")["rider_id"].iloc[0] if len(v) else None)
                                  for m, v in predict.versions(sid)]
                for v in st["versions"]:
                    riders.add(v["fav"])
            out_stages.append(st)
        team = rows.drop_duplicates("rider_id", keep="last").set_index("rider_id")["team"]
        rider_info = {r: [names.get(r, r), team.get(r), prof.loc[r, "type"] if r in prof.index else ""]
                      for r in sorted(x for x in riders if x)}
        status = ("finished" if all(s["status"] == "finished" for s in out_stages)
                  else "running" if any(s["status"] == "finished" for s in out_stages) else "upcoming")
        doc = dict(id=key, slug=race.slug, year=int(race.year), name=cal["name"].get(race.slug, race.name),
                   kind=race.kind, gender=race.gender, start=race.startdate.date().isoformat(), end=str(race.enddate)[:10],
                   status=status, stages=out_stages, riders=rider_info)
        (out / "data" / "race" / f"{key}.json").write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
        nxt = next((s for s in out_stages if s["status"] == "upcoming"), None)
        last = out_stages[-1]
        index.append(dict(id=key, name=doc["name"], kind=race.kind, gender=race.gender, start=doc["start"], end=doc["end"],
                          n=len(out_stages), status=status,
                          early=bool(nxt and nxt.get("provisional")), p_bw=nxt.get("p_bw") if nxt else None,
                          winner=rider_info.get(last.get("result", {}).get("winner"), [None])[0] if status == "finished" and len(out_stages) == 1 else None))

    meta = json.loads((config.MODELS / "metrics.json").read_text())
    lm = live.LiveModel()
    (out / "data" / "index.json").write_text(json.dumps(dict(
        demo=config.is_demo(), today=today.date().isoformat(), season=int(season),
        built=pd.Timestamp.now().isoformat(timespec="minutes"), races=index,
        live=dict(chase=live.CHASE, sd=lm.sd, margin=lm.finale_margin, size_effect=lm.size_effect)),
        separators=(",", ":"), ensure_ascii=False))

    track = pd.read_csv(evaluate.TRACK) if evaluate.TRACK.exists() else pd.DataFrame()
    acc = dict(metrics=meta, calib=[], per_race=[], misses=[])
    if len(track):
        off = track[track["official"]]
        lab = off.dropna(subset=["break_won"])
        b = pd.cut(lab["p_break_wins"], [0, .05, .1, .2, .3, .4, .5, .6, .8, 1])
        acc["calib"] = [dict(pred=_r(g["p_break_wins"].mean()), obs=_r(g["break_won"].mean()), n=int(len(g)))
                        for _, g in lab.groupby(b, observed=True)]
        summ = evaluate.summary(track)
        acc["per_race"] = [dict(race=cal["name"].get(x.slug, x.slug), stages=int(x.stages), brier=_r(x.bw_brier, 3),
                                hits=_r(x.break_hits, 3), top3=_r(x.winner_top3, 3)) for x in summ.itertuples()]
        m = off[off["note"].notna()].sort_values("date", ascending=False).head(15)
        acc["misses"] = [dict(date=x.date, race=cal["name"].get(x.slug, x.slug), id=race_key(x.stage_id.split("/", 1)[1].rsplit("/", 1)[0]),
                              p=_r(x.p_break_wins, 3), note=x.note) for x in m.itertuples()]
        acc["live"] = dict(stages=int(off["stage_id"].nunique()), backtest=int(off["backtest"].sum()) if "backtest" in off else 0)
    (out / "data" / "accuracy.json").write_text(json.dumps(acc, separators=(",", ":"), default=float))
    log(f"site: {len(index)} races of {season} -> {out}")
    return out
