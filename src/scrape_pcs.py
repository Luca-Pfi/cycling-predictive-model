"""Real stage-race results from ProCyclingStats, in the stage-model schema.

Every page is cached as HTML in <out>/cache, so a run can be stopped and resumed.

    # download with the procyclingstats package
    python -m src.scrape_pcs --years 2019-2025

    # or: use race and stage pages you saved from your browser (no downloads at all)
    python -m src.scrape_pcs --import-dir ~/Downloads/pcs --offline

    python -m src.train --data data/pcs
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from procyclingstats import Race, Stage

from src.pcs_pages import PageStore, import_saved_pages

RACES = ["tour-de-france", "giro-d-italia", "vuelta-a-espana", "paris-nice",
         "tirreno-adriatico", "volta-a-catalunya", "itzulia-basque-country",
         "tour-de-romandie", "dauphine", "tour-de-suisse", "tour-de-pologne",
         "uae-tour", "benelux-tour"]
# PCS profile icons: p1 flat, p2 hills/flat finish, p3 hills/uphill finish,
# p4 mountains/flat finish, p5 mountains/uphill finish
ICON = {"p1": ("flat", 0), "p2": ("hilly", 0), "p3": ("hilly", 1),
        "p4": ("mountain", 0), "p5": ("mountain", 1)}
STAGE_URL = re.compile(r"^race/([^/]+)/(\d{4})/stage-\d+[a-z]?$")


def display_name(pcs_name):
    """PCS result tables list 'Surname Firstname' ('Van Aert Wout' -> 'Wout Van Aert')."""
    parts = str(pcs_name).split()
    return " ".join([parts[-1], *parts[:-1]]) if len(parts) > 1 else str(pcs_name)


def parse_stage(s, url):
    """Stage page -> (stage row, result rows), or None for team time trials."""
    if s.stage_type() == "TTT":
        return None
    race, year = STAGE_URL.match(url).groups()
    kind, uphill = ICON.get(s.profile_icon(), ("flat", 0))
    if s.stage_type() == "ITT":
        kind = "itt"
    stage = dict(stage_id=url, race=race, year=int(year), date=s.date(),
                 distance_km=s.distance(), elevation_gain_m=s.vertical_meters(),
                 profile_score=s.profile_score(), uphill_finish=uphill,
                 is_itt=int(kind == "itt"), stage_type=kind)
    rows = [r for r in s.results("rider_name", "rider_url", "team_name", "rank", "status", "age")
            if r.get("status") == "DF" and r.get("rank")]
    return stage, rows


def stage_urls(store, years):
    urls = []
    for year in years:
        for race in RACES:
            try:
                urls += [i["stage_url"] for i in store.get(Race, f"race/{race}/{year}").stages("stage_url")]
            except Exception as exc:  # race not held that year, or not cached offline
                print(f"  skip {race} {year}: {exc}")
    return urls


def main(urls, out, store):
    stages, results, riders = [], [], {}
    for url in urls:
        if not STAGE_URL.match(url):
            continue
        try:
            parsed = parse_stage(store.get(Stage, url), url)
        except Exception as exc:
            print(f"  {url}: {exc}")
            continue
        if not parsed:
            continue
        stage, rows = parsed
        stages.append(stage)
        for r in rows:
            results.append(dict(stage_id=url, rider_id=r["rider_url"], rank=r["rank"]))
            riders[r["rider_url"]] = dict(rider_id=r["rider_url"], name=display_name(r["rider_name"]),
                                          team=r["team_name"],  # latest team wins
                                          birth_year=stage["year"] - (r.get("age") or 27))
        print(f"  {url}: {len(rows)} finishers")
    if not stages:
        raise SystemExit("No stage pages could be read; nothing written.")
    st = pd.DataFrame(stages).dropna(subset=["distance_km"])
    st["elevation_gain_m"] = st["elevation_gain_m"].fillna(st["elevation_gain_m"].median()).fillna(0)
    st["profile_score"] = st["profile_score"].fillna(0)
    st.to_csv(out / "stages.csv", index=False)
    pd.DataFrame(riders.values()).to_csv(out / "riders.csv", index=False)
    pd.DataFrame(results).query("stage_id in @st.stage_id").to_csv(out / "results.csv", index=False)
    print(f"{len(st)} stages, {len(riders)} riders, {len(results)} results -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019-2025", help="e.g. 2022 or 2019-2025")
    ap.add_argument("--import-dir", help="folder of PCS pages saved from your browser")
    ap.add_argument("--offline", action="store_true", help="only use cached / imported pages")
    ap.add_argument("--out", default="data/pcs")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between downloads")
    args = ap.parse_args()
    out = Path(args.out)
    store = PageStore(out / "cache", offline=args.offline, delay=args.delay)
    if args.import_dir:
        import_saved_pages(args.import_dir, store.cache)
    if args.offline:
        urls = store.cached("race_")
    else:
        a, _, b = args.years.partition("-")
        urls = stage_urls(store, range(int(a), int(b or a) + 1))
    main(sorted(set(urls)), out, store)
