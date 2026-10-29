"""Scrape real stage-race results from ProCyclingStats into the project schema.

Uses the `procyclingstats` package. Be polite: keep the delay, scrape only what
you need, and respect the PCS terms of use. Results are cached per stage, so you
can stop and resume.

    python -m src.scrape_pcs --years 2019-2024 --out data/pcs
    python -m src.train --data data/pcs
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from procyclingstats import Race, Stage

RACES = ["tour-de-france", "giro-d-italia", "vuelta-a-espana", "paris-nice",
         "tirreno-adriatico", "volta-a-catalunya", "itzulia-basque-country",
         "tour-de-romandie", "dauphine", "tour-de-suisse", "tour-de-pologne",
         "uae-tour", "benelux-tour"]
# PCS profile icons: p1 flat, p2 hills/flat finish, p3 hills/uphill finish,
# p4 mountains/flat finish, p5 mountains/uphill finish
ICON = {"p1": ("flat", 0), "p2": ("hilly", 0), "p3": ("hilly", 1),
        "p4": ("mountain", 0), "p5": ("mountain", 1)}


def scrape_stage(url, race, year, delay):
    s = Stage(url)
    time.sleep(delay)
    if s.stage_type() == "TTT":
        return None
    kind, uphill = ICON.get(s.profile_icon(), ("flat", 0))
    if s.stage_type() == "ITT":
        kind = "itt"
    stage = dict(stage_id=url, race=race, year=year, date=s.date(),
                 distance_km=s.distance(), elevation_gain_m=s.vertical_meters(),
                 profile_score=s.profile_score(), uphill_finish=uphill,
                 is_itt=int(kind == "itt"), stage_type=kind)
    rows = [r for r in s.results("rider_name", "rider_url", "team_name", "rank", "status", "age")
            if r.get("status") == "DF" and r.get("rank")]
    return dict(stage=stage, results=rows)


def main(years, out, delay):
    out = Path(out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    stages, results, riders = [], [], {}
    for year in years:
        for race in RACES:
            try:
                stage_list = Race(f"race/{race}/{year}").stages("stage_url")
                time.sleep(delay)
            except Exception as exc:  # race not held / page changed
                print(f"skip {race} {year}: {exc}")
                continue
            for item in stage_list:
                url = item["stage_url"]
                f = cache / (url.replace("/", "_") + ".json")
                if f.exists():
                    data = json.loads(f.read_text())
                else:
                    try:
                        data = scrape_stage(url, race, year, delay)
                    except Exception as exc:
                        print(f"  failed {url}: {exc}")
                        continue
                    f.write_text(json.dumps(data))
                    print(f"  {url}")
                if not data:
                    continue
                stages.append(data["stage"])
                for r in data["results"]:
                    results.append(dict(stage_id=url, rider_id=r["rider_url"], rank=r["rank"]))
                    birth = year - (r.get("age") or 27)
                    riders.setdefault(r["rider_url"], dict(rider_id=r["rider_url"],
                                                           name=r["rider_name"],
                                                           team=r["team_name"],
                                                           birth_year=birth))
                    riders[r["rider_url"]]["team"] = r["team_name"]  # keep latest team

    st = pd.DataFrame(stages).dropna(subset=["distance_km"])
    st["elevation_gain_m"] = st["elevation_gain_m"].fillna(st["elevation_gain_m"].median())
    st["profile_score"] = st["profile_score"].fillna(0)
    st.to_csv(out / "stages.csv", index=False)
    pd.DataFrame(riders.values()).to_csv(out / "riders.csv", index=False)
    pd.DataFrame(results).query("stage_id in @st.stage_id").to_csv(out / "results.csv", index=False)
    print(f"{len(st)} stages, {len(riders)} riders, {len(results)} results -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019-2024", help="e.g. 2022 or 2019-2024")
    ap.add_argument("--out", default="data/pcs")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()
    a, _, b = args.years.partition("-")
    main(range(int(a), int(b or a) + 1), args.out, args.delay)
