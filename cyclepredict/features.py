"""Leak-free features for every (stage, rider) pair, past or upcoming.

Every history feature comes from one mechanism: cumulative counters over past
results joined "as of" the stage date (merge_asof, strictly before). The same
code therefore builds training rows for 2021 races and preview rows for next
week's race, and no feature can see the race it describes.
"""
import numpy as np
import pandas as pd

ICON_GROUP = {"p1": "flat", "p2": "hills", "p3": "hills", "p4": "mountain", "p5": "mountain"}
BREAK_PRIOR, WIN_PRIOR = 0.06, 0.01  # shrinkage targets for rider rates
STAGE_FEATURES = [
    "distance_km", "vertical_m", "profile_score", "vert_per_km", "uphill_finish", "icon_n",
    "is_itt", "one_day", "tier", "is_champ", "women", "stage_index", "n_stages", "progress",
    "is_last", "n_climbs", "climbs_hard", "last_climb_n", "log_quality", "month",
    "gc_gap10_log", "race_bw_rate", "race_bw_n", "group_bw_rate", "race_win_solo_share",
    "field_break_mean", "field_break_top30", "n_fast", "n_climbers", "fav_strength",
]
RIDER_FEATURES = [
    "starts", "break_rate", "break_rate_grp", "break_rate_365", "breaks_total",
    "win_rate_grp", "pct_grp", "form20", "race_best_pct", "race_breaks", "race_starts",
    "team_break_rate", "prev_points_log", "season_points_log", "is_pro_team", "age",
    "gc_rank_prev", "gc_gap_prev_log", "in_break_prev", "kom_firsts", "team_rank", "is_leader",
    "home", "days_since",
]


def _asof(left, right, by, cols, tol=None):
    """right's cumulative columns as of each left row's date (strictly before)."""
    left = left.reset_index().sort_values("date")
    right = right.sort_values("date")[by + ["date"] + cols]
    out = pd.merge_asof(left, right, on="date", by=by, allow_exact_matches=False, tolerance=tol)
    return out.set_index("index").reindex(left["index"].sort_values())[cols]


def rider_history(t):
    """Per rider and date, cumulative counters including that day's result."""
    st, res = t["stages"], t["results"]
    r = res.merge(st[["stage_id", "date", "slug", "profile_icon", "stage_type", "labelled", "race_id", "year"]],
                  on="stage_id")
    r = r[r["date"].notna()].copy()
    r["grp"] = np.where(r["stage_type"] == "ITT", "itt", r["profile_icon"].map(ICON_GROUP).fillna("hills"))
    n = r.groupby("stage_id")["rank"].transform("max").clip(lower=2)
    r["pct"] = np.where(r["rank"].notna(), 1 - (r["rank"] - 1) / (n - 1), 0.0)
    r["won"] = (r["rank"] == 1).astype(int)
    r["lab"] = r["labelled"].fillna(False).astype(int)
    r["brk"] = (r["in_break"] & r["labelled"].fillna(False)).astype(int)
    r = r.sort_values(["date", "stage_id"])
    for keys, tag in [(["rider_id"], ""), (["rider_id", "grp"], "_g"), (["rider_id", "slug"], "_r")]:
        g = r.groupby(keys)
        r[f"c_starts{tag}"] = g.cumcount() + 1
        r[f"c_lab{tag}"] = g["lab"].cumsum()
        r[f"c_brk{tag}"] = g["brk"].cumsum()
        r[f"c_win{tag}"] = g["won"].cumsum()
        r[f"c_pct{tag}"] = g["pct"].cumsum()
    r["c_best_r"] = r.groupby(["rider_id", "slug"])["pct"].cummax()
    r["form20"] = r.groupby("rider_id")["pct"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    r["c_pts_season"] = r.groupby(["rider_id", "year"])["pcs_points"].cumsum()
    r["h_year"] = r["year"]
    r["last_date"] = r["date"]
    tg = r.groupby("team")
    r["c_team_lab"], r["c_team_brk"] = tg["lab"].cumsum(), tg["brk"].cumsum()
    return r


def stage_history(st):
    """Per race slug / terrain group, cumulative breakaway outcomes (labelled stages)."""
    s = st[st["labelled"]].copy()
    s["grp"] = s["profile_icon"].map(ICON_GROUP).fillna("hills")
    s["bw"] = s["break_won"].astype(int)
    s["solo"] = (s["win_type"] == "solo").astype(int)
    s = s.sort_values("date")
    for keys, tag in [(["slug"], "r"), (["grp"], "g")]:
        g = s.groupby(keys)
        s[f"{tag}_n"], s[f"{tag}_bw"], s[f"{tag}_solo"] = g.cumcount() + 1, g["bw"].cumsum(), g["solo"].cumsum()
    return s


def starters(t, stage_ids):
    """Who starts each stage: the result list once raced, else the startlist minus abandons."""
    st = t["stages"].set_index("stage_id")
    res = t["results"]
    raced = res[res["stage_id"].isin(stage_ids)][["stage_id", "rider_id", "team"]]
    done = set(raced["stage_id"])
    todo = [s for s in stage_ids if s not in done]
    rows = []
    for sid in todo:
        race_id, k = st.at[sid, "race_id"], st.at[sid, "stage_index"]
        sl = t["startlists"][t["startlists"]["race_id"] == race_id][["rider_id", "team"]]
        prev = st[(st["race_id"] == race_id) & (st["stage_index"] < k) & st["finished"].astype(bool)]
        if len(prev):  # riders who abandoned earlier in the race do not start
            last = res[res["stage_id"] == prev.index[-1]]
            sl = last[last["status"].isin(["DF", "OTL"])][["rider_id", "team"]] if len(last) else sl
        rows.append(sl.assign(stage_id=sid))
    return pd.concat([raced] + rows, ignore_index=True) if rows else raced


def build(t, stage_ids, hist=None, shist=None):
    """Feature rows for every starter of the given stages. Returns (rider_rows, stage_rows)."""
    st = t["stages"].set_index("stage_id")
    hist = rider_history(t) if hist is None else hist
    shist = stage_history(t["stages"]) if shist is None else shist
    q = starters(t, stage_ids)
    q = q.merge(st[["date", "slug", "race_id", "year", "profile_icon", "stage_type", "stage_index", "country"]],
                left_on="stage_id", right_index=True)
    q["grp"] = np.where(q["stage_type"] == "ITT", "itt", q["profile_icon"].map(ICON_GROUP).fillna("hills"))

    a = _asof(q, hist, ["rider_id"], ["c_starts", "c_lab", "c_brk", "c_win", "c_pct", "form20"])
    g = _asof(q, hist, ["rider_id", "grp"], ["c_lab_g", "c_brk_g", "c_win_g", "c_pct_g", "c_starts_g"])
    rr = _asof(q, hist, ["rider_id", "slug"], ["c_starts_r", "c_brk_r", "c_best_r"])
    q365 = q.assign(date=q["date"] - pd.Timedelta(days=365))
    a365 = _asof(q365, hist, ["rider_id"], ["c_lab", "c_brk"])
    f = pd.DataFrame(index=q.index)
    f["starts"] = a["c_starts"].fillna(0)
    lab, brk = a["c_lab"].fillna(0), a["c_brk"].fillna(0)
    f["break_rate"] = (brk + 5 * BREAK_PRIOR) / (lab + 5)
    f["break_rate_grp"] = (g["c_brk_g"].fillna(0) + 5 * BREAK_PRIOR) / (g["c_lab_g"].fillna(0) + 5)
    f["break_rate_365"] = (brk - a365["c_brk"].fillna(0) + 3 * BREAK_PRIOR) / (lab - a365["c_lab"].fillna(0) + 3)
    f["breaks_total"] = brk
    f["win_rate_grp"] = (g["c_win_g"].fillna(0) + 5 * WIN_PRIOR) / (g["c_starts_g"].fillna(0) + 5)
    f["pct_grp"] = (g["c_pct_g"].fillna(0) + 3 * 0.4) / (g["c_starts_g"].fillna(0) + 3)
    f["form20"] = a["form20"].fillna(0.4)
    f["race_best_pct"] = rr["c_best_r"].fillna(-1)
    f["race_breaks"] = rr["c_brk_r"].fillna(0)
    f["race_starts"] = rr["c_starts_r"].fillna(0)
    tb = _asof(q, hist.drop_duplicates(["team", "date", "stage_id"], keep="last"), ["team"], ["c_team_lab", "c_team_brk"])
    f["team_break_rate"] = (tb["c_team_brk"].fillna(0) + 20 * BREAK_PRIOR) / (tb["c_team_lab"].fillna(0) + 20)
    sp = _asof(q, hist, ["rider_id"], ["c_pts_season", "h_year"])
    f["season_points_log"] = np.log1p(np.where(sp["h_year"] == q["year"], sp["c_pts_season"], 0))
    rs = t["rider_seasons"].set_index(["rider_id", "season"])
    key = list(zip(q["rider_id"], q["year"] - 1))
    f["prev_points_log"] = np.log1p(rs["points"].reindex(key).fillna(0).to_numpy())
    cls = rs["team_class"].reindex(list(zip(q["rider_id"], q["year"]))).to_numpy()
    f["is_pro_team"] = (pd.Series(cls, index=q.index) != "wt").astype(float)
    riders = t["riders"].set_index("rider_id")
    born = pd.to_numeric(riders["birthdate"].astype(str).str[:4], errors="coerce")
    f["age"] = q["year"] - q["rider_id"].map(born)
    nat = q["rider_id"].map(riders["nationality"])
    f["home"] = (nat.str.upper() == q["country"].astype(str).str.upper()).astype(float)
    ld = _asof(q, hist, ["rider_id"], ["last_date"])["last_date"]
    f["days_since"] = ((q["date"] - ld).dt.days).fillna(365).clip(upper=365)

    # stage-race context from the previous stage of the same race
    prev = st.reset_index()[["stage_id", "race_id", "stage_index"]]
    prev = prev.assign(stage_index=prev["stage_index"] + 1).rename(columns={"stage_id": "prev_id"})
    q = q.merge(prev, on=["race_id", "stage_index"], how="left").set_index(q.index)
    gc = t["gc"].set_index(["stage_id", "rider_id"]) if len(t["gc"]) else None
    pk = list(zip(q["prev_id"], q["rider_id"]))
    f["gc_rank_prev"] = gc["gc_rank"].reindex(pk).to_numpy() if gc is not None else np.nan
    f["gc_gap_prev_log"] = np.log1p(gc["gc_gap_s"].reindex(pk).to_numpy()) if gc is not None else np.nan
    rb = t["results"].set_index(["stage_id", "rider_id"])["in_break"]
    f["in_break_prev"] = rb.reindex(pk).fillna(False).astype(float).to_numpy()
    # first over a categorised climb earlier in this race (mountains-jersey hunting)
    cl = t["climbs"].merge(st[["race_id", "stage_index"]], left_on="stage_id", right_index=True)
    kom = cl.groupby(["race_id", "first_rider", "stage_index"]).size().rename("n").reset_index()
    kom = kom.rename(columns={"first_rider": "rider_id"}).sort_values("stage_index")
    kom["n"] = kom.groupby(["race_id", "rider_id"])["n"].cumsum()
    kq = q[["race_id", "rider_id", "stage_index"]].reset_index().sort_values("stage_index")
    kq = pd.merge_asof(kq, kom, on="stage_index", by=["race_id", "rider_id"], allow_exact_matches=False)
    f["kom_firsts"] = kq.set_index("index")["n"].reindex(q.index).fillna(0)
    strength = f["prev_points_log"] + f["season_points_log"] + 2 * f["form20"]
    f["team_rank"] = strength.groupby([q["stage_id"], q["team"]]).rank(ascending=False, method="first")
    f["is_leader"] = (f["team_rank"] == 1).astype(float)

    rows = pd.concat([q[["stage_id", "rider_id", "team", "date", "grp"]], f], axis=1)
    stages = stage_rows(t, stage_ids, rows, shist)
    rows = rows.merge(stages, on="stage_id", how="left", suffixes=("", "_s"))
    return rows, stages


def stage_rows(t, stage_ids, rider_rows, shist):
    st = t["stages"].set_index("stage_id").loc[list(stage_ids)].reset_index()
    s = pd.DataFrame({"stage_id": st["stage_id"]})
    s["distance_km"] = st["distance_km"]
    s["vertical_m"] = st["vertical_m"]
    s["profile_score"] = st["profile_score"]
    s["vert_per_km"] = st["vertical_m"] / st["distance_km"].clip(lower=1)
    s["uphill_finish"] = st["profile_icon"].isin(["p3", "p5"]).astype(float)
    s["icon_n"] = st["profile_icon"].astype(str).str[1].apply(pd.to_numeric, errors="coerce")
    s["is_itt"] = (st["stage_type"] == "ITT").astype(float)
    s["one_day"] = st["one_day"].astype(float)
    s["tier"] = st["tier"]
    s["is_champ"] = st["kind"].astype(str).str.startswith("championship").astype(float)
    s["women"] = (st["gender"] == "women").astype(float)
    s["stage_index"], s["n_stages"] = st["stage_index"], st["n_stages"]
    s["progress"] = st["stage_index"] / st["n_stages"]
    s["is_last"] = (st["stage_index"] == st["n_stages"]).astype(float) * (st["n_stages"] > 1)
    s["n_climbs"] = st["n_climbs"]
    s["climbs_hard"] = st[["climbs_hc", "climbs_1"]].fillna(0).sum(axis=1)
    s["last_climb_n"] = st["last_climb_cat"].map({"HC": 5, "1": 4, "2": 3, "3": 2, "4": 1}).fillna(0)
    s["log_quality"] = np.log1p(st["startlist_quality"])
    s["month"] = st["date"].dt.month
    # how tight the GC is before the stage: gap of the 10th rider
    g10 = rider_rows[rider_rows["gc_rank_prev"] == 10].drop_duplicates("stage_id").set_index("stage_id")["gc_gap_prev_log"]
    s["gc_gap10_log"] = s["stage_id"].map(g10)
    q = st[["stage_id", "slug", "date"]].assign(grp=st["profile_icon"].map(ICON_GROUP).fillna("hills"))
    rh = _asof(q, shist, ["slug"], ["r_n", "r_bw", "r_solo"])
    gh = _asof(q, shist, ["grp"], ["g_n", "g_bw"])
    grp_rate = (gh["g_bw"].fillna(0) + 2) / (gh["g_n"].fillna(0) + 10)
    s["group_bw_rate"] = grp_rate.to_numpy()
    s["race_bw_rate"] = ((rh["r_bw"].fillna(0) + 6 * grp_rate) / (rh["r_n"].fillna(0) + 6)).to_numpy()
    s["race_bw_n"] = rh["r_n"].fillna(0).to_numpy()
    s["race_win_solo_share"] = ((rh["r_solo"].fillna(0) + 1) / (rh["r_n"].fillna(0) + 4)).to_numpy()
    by = rider_rows.groupby("stage_id")
    s["field_break_mean"] = s["stage_id"].map(by["break_rate"].mean())
    s["field_break_top30"] = s["stage_id"].map(by["break_rate"].apply(lambda x: x.nlargest(30).sum()))
    fast = rider_rows["win_rate_grp"].where(rider_rows["grp"] == "flat", 0)
    s["n_fast"] = s["stage_id"].map((fast > 0.05).groupby(rider_rows["stage_id"]).sum())
    s["n_climbers"] = s["stage_id"].map(((rider_rows["pct_grp"] > 0.75) & (rider_rows["grp"] == "mountain")).groupby(rider_rows["stage_id"]).sum())
    s["fav_strength"] = s["stage_id"].map(by["win_rate_grp"].apply(lambda x: x.nlargest(3).sum()))
    return s
