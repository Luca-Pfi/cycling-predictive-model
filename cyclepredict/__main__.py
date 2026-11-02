"""Command line.

    python -m cyclepredict demo                 # build the demo seasons (no PCS needed)
    python -m cyclepredict check                # can this computer reach PCS through the package?
    python -m cyclepredict verify-calendar      # check every race slug on PCS
    python -m cyclepredict fetch --years 2019-2025   # history (slow, polite, resumable)
    python -m cyclepredict update               # new results + upcoming profiles/startlists,
                                                # score past previews, retrain, refresh previews
    python -m cyclepredict train | preview | evaluate
    python -m cyclepredict site                 # build the static website into _site/
    python -m cyclepredict import-pages FOLDER  # PCS pages saved from a browser
"""
import argparse
import sys

import pandas as pd

from cyclepredict import config


def _calendar(races=None):
    cal = pd.read_csv(config.CALENDAR)
    return cal[cal["pcs_slug"].isin(races.split(","))] if races else cal


def cmd_check(a):
    from procyclingstats import Rider
    try:
        r = Rider("rider/tadej-pogacar")
        print(f"OK: PCS reachable through the package ({r.name()}, {r.birthdate()})")
    except Exception as exc:
        print(f"PCS refused the request: {exc}\n"
              "Do not try to get around it; save pages from your browser and use `import-pages`.")
        sys.exit(1)


def cmd_verify(a):
    from procyclingstats import Race
    from cyclepredict.pcs import PageStore
    store = PageStore(delay=a.delay)
    cal = pd.read_csv(config.CALENDAR)
    for i, row in cal.iterrows():
        try:
            name = store.get(Race, f"race/{row['pcs_slug']}/{a.year}").name()
            cal.at[i, "slug_status"] = "confirmed"
            print(f"  ok   {row['pcs_slug']:<45} {name}")
        except Exception as exc:
            cal.at[i, "slug_status"] = "not found"
            print(f"  FAIL {row['pcs_slug']:<45} {str(exc)[:60]}")
    cal.to_csv(config.CALENDAR, index=False)


def _fetch(years, races, refresh_unfinished, delay, offline=False):
    from cyclepredict.pcs import PageStore, Tables, fetch_edition, fetch_riders
    store, tables = PageStore(delay=delay, offline=offline), Tables()
    known = pd.read_csv(config.REAL / "races.csv") if (config.REAL / "races.csv").exists() else pd.DataFrame()
    known = known.set_index("race_id") if len(known) else known
    today = pd.Timestamp.today().normalize()
    for year in years:
        for _, row in _calendar(races).iterrows():
            if row["pcs_slug"].startswith("olympic") and year % 4:
                continue
            rid = f"{row['pcs_slug']}/{year}"
            if refresh_unfinished and rid in known.index:
                start, end = pd.to_datetime(known.at[rid, "startdate"]), pd.to_datetime(known.at[rid, "enddate"])
                if end < today - pd.Timedelta(days=3) or start > today + pd.Timedelta(days=45):
                    continue  # finished and stored, or too far ahead to have a startlist
            try:
                fetch_edition(store, tables, row, year, refresh_unfinished)
                print(f"  {row['pcs_slug']} {year}")
            except Exception as exc:
                print(f"  {row['pcs_slug']} {year}: {exc}")
            tables.save()
    real = config.REAL  # riders seen in results or startlists but not fetched yet
    seen = set()
    for f in ["results.csv", "startlists.csv"]:
        if (real / f).exists():
            seen |= set(pd.read_csv(real / f, usecols=["rider_id"])["rider_id"].dropna())
    have = set(pd.read_csv(real / "riders.csv")["rider_id"]) if (real / "riders.csv").exists() else set()
    todo = sorted(seen - have)
    print(f"riders to fetch: {len(todo)}")
    fetch_riders(store, tables, todo)
    tables.save()


def cmd_fetch(a):
    lo, _, hi = a.years.partition("-")
    _fetch(range(int(lo), int(hi or lo) + 1), a.races, False, a.delay, a.offline)


def cmd_update(a):
    from cyclepredict import evaluate, models, predict
    year = pd.Timestamp.today().year
    _fetch([year] if pd.Timestamp.today().month > 1 else [year - 1, year], a.races, True, a.delay, a.offline)
    evaluate.evaluate()
    models.train()
    predict.refresh()


def main():
    ap = argparse.ArgumentParser(prog="cyclepredict")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("check")
    v = sub.add_parser("verify-calendar")
    v.add_argument("--year", type=int, default=pd.Timestamp.today().year - 1)
    v.add_argument("--delay", type=float, default=2.0)
    for name in ["fetch", "update"]:
        p = sub.add_parser(name)
        if name == "fetch":
            p.add_argument("--years", required=True)
        p.add_argument("--races", help="comma-separated PCS slugs (default: whole calendar)")
        p.add_argument("--delay", type=float, default=2.0, help="seconds between PCS requests")
        p.add_argument("--offline", action="store_true", help="only use cached / imported pages")
    sub.add_parser("train")
    sub.add_parser("preview")
    sub.add_parser("evaluate")
    sp = sub.add_parser("site")
    sp.add_argument("--out", default="_site")
    ip = sub.add_parser("import-pages")
    ip.add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "demo":
        from cyclepredict import demo
        demo.write()
    elif a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "verify-calendar":
        cmd_verify(a)
    elif a.cmd == "fetch":
        cmd_fetch(a)
    elif a.cmd == "update":
        cmd_update(a)
    elif a.cmd == "train":
        from cyclepredict import models
        models.train()
    elif a.cmd == "preview":
        from cyclepredict import predict
        predict.refresh()
    elif a.cmd == "evaluate":
        from cyclepredict import evaluate
        evaluate.evaluate()
    elif a.cmd == "site":
        from cyclepredict import site
        site.build(a.out)
    elif a.cmd == "import-pages":
        from cyclepredict.pcs import PageStore
        print(f"imported {PageStore(offline=True).import_saved(a.folder)} pages")


if __name__ == "__main__":
    main()
