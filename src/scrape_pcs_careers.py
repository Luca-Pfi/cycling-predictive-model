"""Scrape real rider careers from ProCyclingStats into the career schema.

PCS rider pages give birthdate, nationality, height, weight, PCS points per
season and team history (with the team's class). Lab tests, junior and U23
results are not on PCS, so those columns stay empty and the model simply
trains without them (`career_model.available` skips empty columns).

Riders come from the season rankings of the chosen years (or from --riders,
a text file with one PCS rider URL per line, e.g. "rider/tadej-pogacar").
Pages are cached in data/pcs_careers/cache, so you can stop and resume.

    python -m src.scrape_pcs_careers --years 2005-2025 --top 300 --out data/pcs_careers
    python -m src.train_career --data data/pcs_careers --models models/career_pcs
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from procyclingstats import Ranking, Rider

CLASS = {"WT": "wt", "PRT": "pro", "PR": "pro", "PCT": "pro", "CT": "conti"}


def ranked_riders(years, top, delay):
    urls = set()
    for y in years:
        for page in range(0, top, 100):
            try:
                rk = Ranking(f"rankings/me/season-individual?date={y}-12-31&offset={page}")
                urls |= {r["rider_url"] for r in rk.individual_ranking("rider_url")}
            except Exception as exc:
                print(f"ranking {y} offset {page} failed: {exc}")
            time.sleep(delay)
    return sorted(urls)


def scrape_rider(url):
    r = Rider(url)
    teams = r.teams_history("season", "class")
    return dict(url=url, name=r.name(), birthdate=r.birthdate(), nationality=r.nationality(),
                height=r.height(), weight=r.weight(),
                points=r.points_per_season_history("season", "points"),
                teams=[{"season": t["season"], "class": t.get("class")} for t in teams])


def main(urls, out, delay):
    out = Path(out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    riders, seasons = [], []
    for i, url in enumerate(urls):
        f = cache / (url.replace("/", "_") + ".json")
        if f.exists():
            data = json.loads(f.read_text())
        else:
            try:
                data = scrape_rider(url)
            except Exception as exc:
                print(f"  failed {url}: {exc}")
                continue
            f.write_text(json.dumps(data))
            time.sleep(delay)
        if not data.get("birthdate"):
            continue
        by = int(str(data["birthdate"])[:4])
        riders.append(dict(rider_id=url, name=data["name"], nationality=data["nationality"],
                           birth_year=by, height_cm=(data["height"] or 0) * 100 or None,
                           weight_kg=data["weight"]))
        level = {int(t["season"]): CLASS.get(t["class"], "amateur") for t in data["teams"] if t.get("season")}
        for p in data["points"]:
            season = int(p["season"])
            seasons.append(dict(rider_id=url, season=season, age=season - by,
                                points=float(p["points"] or 0), team_level=level.get(season, "amateur")))
        if i % 50 == 0:
            print(f"{i}/{len(urls)} riders")
    s = pd.DataFrame(seasons)
    s = s[(s["age"] >= 19) & (s["age"] <= 38)]
    pd.DataFrame(riders).to_csv(out / "riders.csv", index=False)
    s.to_csv(out / "seasons.csv", index=False)
    print(f"{len(riders)} riders, {len(s)} rider-seasons -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2005-2025")
    ap.add_argument("--top", type=int, default=300, help="riders per season ranking")
    ap.add_argument("--riders", help="optional file with one PCS rider URL per line")
    ap.add_argument("--out", default="data/pcs_careers")
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    a, _, b = args.years.partition("-")
    urls = (Path(args.riders).read_text().split() if args.riders
            else ranked_riders(range(int(a), int(b or a) + 1), args.top, args.delay))
    main(urls, args.out, args.delay)
