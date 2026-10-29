"""Build the two real-data datasets from open sources (no scraping).

Sources (all real riders, openly licensed, cloned from GitHub):
  * TidyTuesday 2020-04-07 Tour de France data (CC0): every stage result
    1903-2017 with rider, age, team and rank, plus stage date, distance, type.
    https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-04-07
  * LeTourDataSet (MIT): final general classification of every Tour, 1903-2025.
    https://github.com/thomascamminady/LeTourDataSet
  * josselingirault/procyclingstats.com-webscraper (MIT): birthdate, country,
    height and weight of the 1,005 riders in 2018 PCS teams.
    https://github.com/josselingirault/procyclingstats.com-webscraper

Outputs (same schemas as the synthetic data, so the same models and app work):
  data/tdf_stages/   stages.csv, riders.csv, results.csv  -> stage page
  data/tdf_careers/  riders.csv, seasons.csv              -> career page

"Points" are computed from the real ranks with one fixed scale, documented below.

    python -m src.import_open_data --sources /path/to/clones --from-year 1985
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# points for stage places 1-20 and final GC places 1-20 (this project's own scale)
STAGE_POINTS = [100, 80, 65, 55, 45, 35, 30, 25, 20, 17, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
GC_POINTS = [600, 450, 375, 300, 250, 210, 180, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25]
TYPES = {  # TidyTuesday stage type -> (stage_type, is_itt, is_hilly, is_mountain)
    "flat": ("flat", 0, 0, 0), "hilly": ("hilly", 0, 1, 0),
    "mountain": ("mountain", 0, 0, 1), "itt": ("itt", 1, 0, 0),
}
COUNTRY = {"France": "FRA", "Belgium": "BEL", "Italy": "ITA", "Spain": "ESP", "Netherlands": "NED",
           "Germany": "GER", "Great Britain": "GBR", "Denmark": "DEN", "Slovenia": "SLO",
           "Colombia": "COL", "United States": "USA", "Australia": "AUS", "Norway": "NOR",
           "Switzerland": "SUI"}


def stage_kind(t):
    t = str(t).lower()
    if "team time trial" in t:
        return None
    if "time trial" in t:
        return "itt"
    if "medium mountain" in t or "hilly" in t:
        return "hilly"
    if "mountain" in t:
        return "mountain"
    return "flat"  # plain, flat, transition, intermediate, cobbles, half stage


def name_key(name):
    """Accent/case/order-insensitive key: 'Thomas Geraint' == 'GERAINT THOMAS'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return "-".join(sorted(re.findall(r"[a-z]+", s)))


def flip(pcs_name):
    """'Van Avermaet Greg' -> 'Greg Van Avermaet'."""
    parts = str(pcs_name).split()
    return " ".join([parts[-1], *parts[:-1]]) if len(parts) > 1 else str(pcs_name)


def load_results(src, from_year):
    d = Path(src) / "tidytuesday/data/2020/2020-04-07"
    res = pd.read_csv(d / "stage_data.csv", low_memory=False)
    stg = pd.read_csv(d / "tdf_stages.csv")
    res = res[(res["year"] >= from_year) & res["rank"].astype(str).str.isnumeric() & res["age"].notna()].copy()
    res["rank"] = res["rank"].astype(int)
    res["stage"] = res["stage_results_id"].str.replace("stage-", "").replace({"0": "P"})
    stg["year"] = pd.to_datetime(stg["Date"]).dt.year
    stg["kind"] = stg["Type"].map(stage_kind)
    stg = stg[(stg["year"] >= from_year) & stg["kind"].notna()]
    stg["stage_id"] = "tdf-" + stg["year"].astype(str) + "-" + stg["Stage"].astype(str)
    res["stage_id"] = "tdf-" + res["year"].astype(str) + "-" + res["stage"]
    res = res[res["stage_id"].isin(stg["stage_id"])]
    # the source keeps some annulled results (e.g. 2006/2007 stage wins) next to the
    # reassigned ones, plus a few duplicate rows: keep one row per rider and give
    # every finisher a unique place, in the source's order
    res = res.drop_duplicates(["stage_id", "rider"]).sort_values(["stage_id", "rank"], kind="stable")
    res["rank"] = res.groupby("stage_id").cumcount() + 1
    return res, stg


def assign_riders(res):
    """One id per real person: name key + birth-year cluster (namesakes in other eras)."""
    res["key"] = res["rider"].map(name_key)
    res["by"] = res["year"] - res["age"]
    ids = {}
    for key, g in res.groupby("key"):
        years = np.sort(g["by"].unique())
        cluster, start = {}, years[0]
        for y in years:
            if y - start > 2:
                start = y
            cluster[y] = start
        for y, c in cluster.items():
            ids[(key, y)] = f"{key}-{int(c)}"
    res["rider_id"] = [ids[(k, y)] for k, y in zip(res["key"], res["by"])]
    return res


def body_data(src):
    r = pd.read_csv(Path(src) / "jg/Datasets/RidersInfo2018.csv")
    r["key"] = r["Fullname"].map(name_key)
    r["birth_year_pcs"] = (r["Birthdate"] // 10000).astype(int)
    r["height_cm"] = (r["Height"] * 100).where(r["Height"] > 0).round()
    r["weight_kg"] = r["Weight"].where(r["Weight"] > 0)
    r["nationality"] = r["Country"].map(lambda c: COUNTRY.get(c, "OTHER"))
    return r[["key", "birth_year_pcs", "height_cm", "weight_kg", "nationality"]].drop_duplicates("key")


def gc_ranks(src, from_year):
    g = pd.read_csv(Path(src) / "letourdataset/data/men/TDF_Riders_History.csv")
    g = g[(g["Year"] >= from_year) & pd.to_numeric(g["Rank"], errors="coerce").notna()]
    g = g.assign(key=g["Rider"].map(name_key), gc_rank=g["Rank"].astype(int), year=g["Year"])
    return g[["key", "year", "gc_rank"]].drop_duplicates(["key", "year"])


def build(src, out_root, from_year):
    res, stg = load_results(src, from_year)
    res = assign_riders(res)
    body = body_data(src)

    # ---- riders (shared by both datasets)
    last = res.sort_values("year").groupby("rider_id").last()
    riders = pd.DataFrame({
        "rider_id": last.index, "key": last["key"].values,
        "name": [flip(n) for n in last["rider"]],
        "team": last["team"].fillna("–").values,
        "birth_year": res.groupby("rider_id")["by"].agg(lambda x: int(x.mode().iloc[0])).reindex(last.index).values,
    })
    riders = riders.merge(body, on="key", how="left")
    # only trust the 2018 body data when the birth year agrees (same person)
    same = (riders["birth_year_pcs"] - riders["birth_year"]).abs() <= 1
    for c in ["height_cm", "weight_kg", "nationality"]:
        riders.loc[~same, c] = np.nan

    # ---- stage dataset
    out = Path(out_root) / "tdf_stages"
    out.mkdir(parents=True, exist_ok=True)
    kinds = stg["kind"].map(TYPES)
    stages = pd.DataFrame({
        "stage_id": stg["stage_id"], "race": "Tour de France", "year": stg["year"],
        "date": pd.to_datetime(stg["Date"]).dt.date, "distance_km": stg["Distance"],
        "is_itt": [k[1] for k in kinds], "is_hilly": [k[2] for k in kinds],
        "is_mountain": [k[3] for k in kinds], "stage_type": [k[0] for k in kinds],
    })
    stages.to_csv(out / "stages.csv", index=False)
    riders[["rider_id", "name", "team", "birth_year"]].to_csv(out / "riders.csv", index=False)
    res[["stage_id", "rider_id", "rank"]].to_csv(out / "results.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "title": f"Tour de France {from_year}-{int(stages['year'].max())} (real results)",
        "sources": ["TidyTuesday 2020-04-07 (CC0)"]}, indent=2))
    print(f"stages: {len(stages)} stages, {res['rider_id'].nunique()} riders, {len(res)} results -> {out}")

    # ---- career dataset: one "season" = one Tour de France
    out = Path(out_root) / "tdf_careers"
    out.mkdir(parents=True, exist_ok=True)
    scale = dict(enumerate(STAGE_POINTS, 1))
    res["stage_pts"] = res["rank"].map(scale).fillna(0)
    seasons = res.groupby(["rider_id", "year"]).agg(
        age=("age", "max"), stage_points=("stage_pts", "sum"), race_days=("rank", "size"),
        wins=("rank", lambda r: int((r == 1).sum())), top10s=("rank", lambda r: int((r <= 10).sum())),
        key=("key", "first")).reset_index()
    seasons = seasons.merge(gc_ranks(src, from_year), on=["key", "year"], how="left")
    seasons["gc_points"] = seasons["gc_rank"].map(dict(enumerate(GC_POINTS, 1))).fillna(0)
    seasons["points"] = seasons["stage_points"] + seasons["gc_points"]
    seasons = seasons.rename(columns={"year": "season"})
    # one consistent age per Tour (the source's per-row ages can repeat across years)
    seasons["age"] = seasons["season"] - seasons["rider_id"].map(riders.set_index("rider_id")["birth_year"])
    seasons = seasons[(seasons["age"] >= 19) & (seasons["age"] <= 38)]
    seasons[["rider_id", "season", "age", "points", "race_days", "wins", "top10s", "gc_rank"]].to_csv(
        out / "seasons.csv", index=False)
    riders[["rider_id", "name", "nationality", "birth_year", "height_cm", "weight_kg"]].to_csv(
        out / "riders.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "title": f"Tour de France careers {from_year}-{int(seasons['season'].max())} (real riders)",
        "points_label": "Tour de France points",
        "season_word": "Tour",
        "active_word": "still riding the Tour",
        "points_note": "Points per Tour from real results: stage places 1-20 score "
                       f"{STAGE_POINTS[0]}-{STAGE_POINTS[-1]}, final GC places 1-20 score "
                       f"{GC_POINTS[0]}-{GC_POINTS[-1]}.",
        "sources": ["TidyTuesday 2020-04-07 (CC0)", "LeTourDataSet (MIT)",
                    "josselingirault/procyclingstats.com-webscraper (MIT)"]}, indent=2))
    matched = riders["height_cm"].notna().sum()
    print(f"careers: {seasons['rider_id'].nunique()} riders, {len(seasons)} Tours, "
          f"{matched} riders with height/weight -> {out}")
    print("riders per age:", seasons.groupby("age").size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True,
                    help="folder containing clones named tidytuesday/, letourdataset/, jg/")
    ap.add_argument("--out", default="data")
    ap.add_argument("--from-year", type=int, default=1985)
    args = ap.parse_args()
    build(args.sources, args.out, args.from_year)
