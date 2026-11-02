"""Load the tables (real or demo) and add the labels the models learn from."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cyclepredict import config

TABLES = ["races", "stages", "results", "gc", "climbs", "startlists", "riders", "rider_seasons"]


def read(folder, name):
    for f in (Path(folder) / f"{name}.csv", Path(folder) / f"{name}.csv.gz"):
        if f.exists():
            return pd.read_csv(f, low_memory=False)
    return pd.DataFrame()


def today(folder=None):
    """The data's clock: the real date, or the demo's fixed date."""
    f = Path(folder or config.data_dir()) / "DEMO_TODAY"
    return pd.Timestamp(f.read_text().strip()) if f.exists() else pd.Timestamp.today().normalize()


def solo_km(won_how):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*km solo", str(won_how or ""))
    return float(m.group(1)) if m else np.nan


def win_type(won_how):
    w = str(won_how or "").lower()
    if "solo" in w:
        return "solo"
    if "large group" in w:
        return "bunch sprint"
    if "small group" in w or "sprint of" in w:
        return "small group"
    if "time trial" in w:
        return "time trial"
    return "unknown"


def load(folder=None):
    folder = Path(folder or config.data_dir())
    t = {k: read(folder, k) for k in TABLES}
    cal = pd.read_csv(config.CALENDAR)
    races = t["races"].merge(cal[["pcs_slug", "terrain", "country", "name"]].rename(
        columns={"pcs_slug": "slug", "name": "cal_name"}), on="slug", how="left")
    races["tier"] = np.where(races["uci_tour"].astype(str).str.contains("UWT|WWT"), 2, 1)
    races["startdate"] = pd.to_datetime(races["startdate"])
    t["races"] = races
    st = t["stages"].merge(races[["race_id", "name", "gender", "kind", "tier", "terrain", "country"]], on="race_id", how="left")
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values(["date", "stage_id"]).reset_index(drop=True)
    st["n_stages"] = st.groupby("race_id")["stage_id"].transform("size")
    st["stage_index"] = st.groupby("race_id").cumcount() + 1
    st["one_day"] = st["n_stages"] == 1
    st["win_type"] = st["won_how"].map(win_type)
    st["solo_km"] = st["won_how"].map(solo_km)
    t["stages"] = st
    return t, folder


def label_breaks(t):
    """Who was in the day's breakaway and whether it won, from PCS breakaway km.

    In the break: breakaway_kms >= max(BREAK_MIN_KM, BREAK_MIN_SHARE * distance).
    A race-year counts only if PCS recorded breakaway km for at least one of its
    stages; otherwise "nobody had breakaway km" means missing data, not no break.
    """
    st, res = t["stages"], t["results"]
    res = res.merge(st[["stage_id", "race_id", "distance_km", "stage_type", "finished"]], on="stage_id")
    thr = np.maximum(config.BREAK_MIN_KM, config.BREAK_MIN_SHARE * res["distance_km"].fillna(180))
    res["in_break"] = res["breakaway_kms"].fillna(0) >= thr
    covered = res.groupby("race_id")["breakaway_kms"].max().fillna(0) > 0
    per = res.groupby("stage_id").agg(break_size=("in_break", "sum"))
    winners = res[res["rank"] == 1].drop_duplicates("stage_id").set_index("stage_id")
    per["winner"] = winners["rider_id"]
    per["break_won"] = winners["in_break"].reindex(per.index).fillna(False).astype(bool)
    st = st.drop(columns=[c for c in ["break_size", "winner", "break_won", "labelled"] if c in st]).merge(
        per, left_on="stage_id", right_index=True, how="left")
    st["labelled"] = (st["finished"].astype(bool) & st["race_id"].map(covered).fillna(False)
                      & (st["stage_type"] == "RR") & st["winner"].notna())
    t["stages"], t["results"] = st, res.drop(columns=["race_id", "distance_km", "stage_type", "finished"])
    return t
