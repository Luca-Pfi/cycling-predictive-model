"""Real rider careers from ProCyclingStats, in the career-model schema.

PCS rider pages give birthdate, nationality, height, weight, PCS points per
season and team history (with the team's class). Lab tests, junior and U23
results are not on PCS, so those columns stay empty and the model trains
without them (`career_model.available` skips empty columns).

Which riders: the top `--top` of each season ranking in `--years`, or the
PCS URLs listed in a `--riders` file (one per line, e.g. rider/tadej-pogacar).
Every page is cached as HTML in <out>/cache, so a run can be stopped and resumed.

    # download with the procyclingstats package
    python -m src.scrape_pcs_careers --years 2010-2025 --top 300

    # or: use rider pages you saved from your browser (no downloads at all)
    python -m src.scrape_pcs_careers --import-dir ~/Downloads/pcs --offline

    # then train; the app shows these riders afterwards
    python -m src.train_career --data data/pcs_careers
"""
import argparse
from pathlib import Path

import pandas as pd
from procyclingstats import Ranking, Rider

from src.pcs_pages import PageStore, import_saved_pages

CLASS = {"WT": "wt", "PRT": "pro", "PR": "pro", "PCT": "pro", "CT": "conti"}


def ranked_riders(store, years, top):
    urls = set()
    for y in years:
        for offset in range(0, top, 100):
            url = f"rankings/me/season-individual?date={y}-12-31&offset={offset}"
            try:
                urls |= {r["rider_url"] for r in store.get(Ranking, url).individual_ranking("rider_url")}
            except Exception as exc:
                print(f"  ranking {y} offset {offset}: {exc}")
    return sorted(urls)


def parse_rider(r, url):
    """Rider page -> (rider row, season rows)."""
    birth = r.birthdate()
    if not birth:
        return None, []
    by = int(str(birth)[:4])
    height = r.height()
    rider = dict(rider_id=url, name=r.name(), nationality=r.nationality(), birth_year=by,
                 height_cm=round(height * 100) if height else None, weight_kg=r.weight())
    level = {}
    for t in r.teams_history("season", "class"):
        if t.get("season"):  # a rider can change team mid-season: keep the highest level
            new = CLASS.get(t.get("class"), "amateur")
            old = level.get(int(t["season"]), "amateur")
            level[int(t["season"])] = max(old, new, key=["amateur", "conti", "pro", "wt"].index)
    seasons = [dict(rider_id=url, season=int(p["season"]), age=int(p["season"]) - by,
                    points=float(p["points"] or 0), team_level=level.get(int(p["season"]), "amateur"))
               for p in r.points_per_season_history("season", "points")]
    return rider, seasons


def main(urls, out, store):
    riders, seasons = [], []
    for i, url in enumerate(urls, 1):
        try:
            rider, rows = parse_rider(store.get(Rider, url), url)
        except Exception as exc:
            print(f"  {url}: {exc}")
            continue
        if rider:
            riders.append(rider)
            seasons += rows
        if i % 50 == 0:
            print(f"{i}/{len(urls)} riders")
    if not riders:
        raise SystemExit("No rider pages could be read; nothing written.")
    s = pd.DataFrame(seasons)
    s = s[(s["age"] >= 19) & (s["age"] <= 38)]
    pd.DataFrame(riders).to_csv(out / "riders.csv", index=False)
    s.to_csv(out / "seasons.csv", index=False)
    print(f"{len(riders)} riders, {len(s)} rider-seasons -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2010-2025")
    ap.add_argument("--top", type=int, default=300, help="riders per season ranking")
    ap.add_argument("--riders", help="file with one PCS rider URL per line")
    ap.add_argument("--import-dir", help="folder of PCS pages saved from your browser")
    ap.add_argument("--offline", action="store_true", help="only use cached / imported pages")
    ap.add_argument("--out", default="data/pcs_careers")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between downloads")
    args = ap.parse_args()
    out = Path(args.out)
    store = PageStore(out / "cache", offline=args.offline, delay=args.delay)
    if args.import_dir:
        import_saved_pages(args.import_dir, store.cache)
    if args.riders:
        urls = Path(args.riders).read_text().split()
    elif args.offline:
        urls = store.cached("rider_")
    else:
        a, _, b = args.years.partition("-")
        urls = ranked_riders(store, range(int(a), int(b or a) + 1), args.top)
    main(urls, out, store)
