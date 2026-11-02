"""ProCyclingStats data through the `procyclingstats` package.

Pages are downloaded by the package with normal, polite requests and kept as raw
HTML in data/pcs_pages, so each page is fetched once and can be re-parsed. Pages
saved from a browser can be imported instead (recognised by their canonical link).
If PCS refuses automated requests ("Cloudflare protection detected"), stop: that
is PCS saying no, and the import route is the alternative.

Parsed tables (data/real/*.csv):
  races      race_id, slug, year, name, gender, kind, uci_tour, startdate, enddate
  stages     stage_id, race_id, slug, year, stage_number, date, distance_km, vertical_m,
             profile_score, profile_icon, stage_type, departure, arrival, won_how,
             startlist_quality, n_climbs, climbs_hc, climbs_1, climbs_2, climbs_3, climbs_4,
             last_climb, last_climb_cat, finished
  results    stage_id, rider_id, team, rank, status, age, breakaway_kms, pcs_points
  gc         stage_id, rider_id, gc_rank, gc_gap_s
  climbs     stage_id, order, climb, category, first_rider
  startlists race_id, rider_id, team, rider_number
  riders     rider_id, name, nationality, birthdate, height, weight, spec_* (6 specialties)
  rider_seasons rider_id, season, points, rank, team_class
"""
import re
import shutil
import time
from pathlib import Path

import pandas as pd

from cyclepredict import config

CANONICAL = re.compile(r'<link rel="canonical" href="https?://(?:www\.)?procyclingstats\.com/([^"?#]+)"')
SPECIALTIES = ["one_day_races", "gc", "time_trial", "sprint", "climber", "hills"]
CLASS = {"WT": "wt", "PRT": "pro", "PR": "pro", "PCT": "pro", "CT": "conti"}
KEYS = {"races": ["race_id"], "stages": ["stage_id"], "results": ["stage_id", "rider_id"],
        "gc": ["stage_id", "rider_id"], "climbs": ["stage_id", "order"],
        "startlists": ["race_id", "rider_id"], "riders": ["rider_id"],
        "rider_seasons": ["rider_id", "season"]}


# ------------------------------------------------------------------ page store
def normalise(url):
    url = re.sub(r"^https?://(www\.)?procyclingstats\.com/", "", url.strip()).strip("/")
    url = re.sub(r"(/result)+$", "/result", url)        # one-day result pages
    url = re.sub(r"(stage-\d+[a-z]?)(/result)+$", r"\1", url)
    url = re.sub(r"(/startlist)+$", "/startlist", url)
    url = re.sub(r"/overview.*$", "", url)
    return url


class PageStore:
    def __init__(self, cache=config.PAGES, offline=False, delay=2.0):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline, self.delay = offline, delay

    def path(self, url):
        return self.cache / (re.sub(r"[^\w.-]", "_", normalise(url)) + ".html")

    def get(self, cls, url, refresh=False):
        url = normalise(url)
        f = self.path(url)
        if f.exists() and not refresh:
            return cls(url, html=f.read_text(encoding="utf-8"), update_html=False)
        if self.offline:
            raise FileNotFoundError(f"not cached (offline): {url}")
        obj = cls(url)  # normal download by the package
        f.write_text(obj.html.html, encoding="utf-8")
        time.sleep(self.delay)
        return obj

    def import_saved(self, folder):
        n = 0
        for f in sorted(Path(folder).rglob("*")):
            if f.is_file() and f.suffix.lower() in {".html", ".htm", ".txt"}:
                m = CANONICAL.search(f.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    shutil.copyfile(f, self.path(m.group(1)))
                    n += 1
        return n


# ---------------------------------------------------------------------- parse
def _try(fn, *a, default=None):
    try:
        v = fn(*a)
        return default if v in ("", []) else v
    except Exception:
        return default


def rider_id(url):
    return str(url or "").replace("rider/", "").strip("/") or None


def _seconds(t):
    if not t or not isinstance(t, str) or ":" not in t:
        return None
    parts = [int(p) for p in t.strip().split(":") if p.isdigit()]
    s = 0
    for p in parts:
        s = s * 60 + p
    return s


def parse_stage(s, race_id, slug, year, stage_number):
    """Stage (or one-day result) page -> dict of table rows. Works before the race too."""
    url = s.relative_url()
    climbs = _try(s.climbs) or []
    cats = [str(c.get("category") or "").upper() for c in climbs]
    results = _try(s.results, "rider_url", "team_name", "rank", "status", "age",
                   "breakaway_kms", "pcs_points") or []
    stage = dict(
        stage_id=url, race_id=race_id, slug=slug, year=year, stage_number=stage_number,
        date=_try(s.date), distance_km=_try(s.distance), vertical_m=_try(s.vertical_meters),
        profile_score=_try(s.profile_score), profile_icon=_try(s.profile_icon),
        stage_type=_try(s.stage_type), departure=_try(s.departure), arrival=_try(s.arrival),
        won_how=_try(s.won_how), startlist_quality=(_try(s.race_startlist_quality_score) or [None])[0],
        n_climbs=len(climbs), climbs_hc=cats.count("HC"), climbs_1=cats.count("1"),
        climbs_2=cats.count("2"), climbs_3=cats.count("3"), climbs_4=cats.count("4"),
        last_climb=climbs[-1].get("climb_name") if climbs else None,
        last_climb_cat=cats[-1] if climbs else None,
        finished=bool(results),
    )
    res = [dict(stage_id=url, rider_id=rider_id(r.get("rider_url")), team=r.get("team_name"),
                rank=r.get("rank"), status=r.get("status"), age=r.get("age"),
                breakaway_kms=r.get("breakaway_kms") or 0, pcs_points=r.get("pcs_points"))
           for r in results if r.get("rider_url")]
    gc_rows = _try(s.gc, "rider_url", "rank", "time") or []
    lead = _seconds(gc_rows[0].get("time")) if gc_rows else None
    gc = [dict(stage_id=url, rider_id=rider_id(g["rider_url"]), gc_rank=g.get("rank"),
               gc_gap_s=(_seconds(g.get("time")) - lead) if lead is not None and _seconds(g.get("time")) else None)
          for g in gc_rows if g.get("rider_url")]
    cl = [dict(stage_id=url, order=i, climb=c.get("climb_name"), category=c.get("category"),
               first_rider=rider_id((c.get("rank") or [{}])[0].get("rider_url")))
          for i, c in enumerate(climbs)]
    return dict(stages=[stage], results=res, gc=gc, climbs=cl)


def parse_rider(r):
    rid = rider_id(r.relative_url())
    spec = _try(r.points_per_speciality) or {}
    height = _try(r.height)
    row = dict(rider_id=rid, name=_try(r.name), nationality=_try(r.nationality),
               birthdate=_try(r.birthdate), height=height, weight=_try(r.weight),
               **{f"spec_{k}": spec.get(k) for k in SPECIALTIES})
    classes = {}
    for t in _try(r.teams_history, "season", "class") or []:
        if t.get("season"):
            classes[int(t["season"])] = CLASS.get(t.get("class"), "other")
    seasons = [dict(rider_id=rid, season=int(p["season"]), points=p.get("points"), rank=p.get("rank"),
                    team_class=classes.get(int(p["season"])))
               for p in _try(r.points_per_season_history, "season", "points", "rank") or []]
    return dict(riders=[row], rider_seasons=seasons)


# ---------------------------------------------------------------------- tables
class Tables:
    """Accumulates parsed rows and upserts them into data/real/*.csv."""

    def __init__(self, folder=config.REAL):
        self.folder = Path(folder)
        self.rows = {k: [] for k in KEYS}

    def add(self, parsed):
        for k, rows in parsed.items():
            self.rows[k] += rows

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        for k, rows in self.rows.items():
            if not rows:
                continue
            new = pd.DataFrame(rows)
            f = self.folder / f"{k}.csv"
            if f.exists():
                old = pd.read_csv(f, low_memory=False)
                if k in ("results", "gc", "climbs"):  # a re-parsed stage replaces its rows
                    old = old[~old["stage_id"].isin(new["stage_id"])]
                new = pd.concat([old, new], ignore_index=True)
            new.drop_duplicates(KEYS[k], keep="last").to_csv(f, index=False)
        self.rows = {k: [] for k in KEYS}


# ---------------------------------------------------------------------- fetch
def race_pages(store, slug, year, kind, gender, refresh=False):
    """(race row, [(stage_url, stage_number)]) for one edition."""
    from procyclingstats import Race
    race = store.get(Race, f"race/{slug}/{year}", refresh=refresh)
    race_id = f"{slug}/{year}"
    row = dict(race_id=race_id, slug=slug, year=year, name=_try(race.name), gender=gender, kind=kind,
               uci_tour=_try(race.uci_tour), startdate=_try(race.startdate), enddate=_try(race.enddate))
    if _try(race.is_one_day_race, default=True):
        return row, [(f"race/{slug}/{year}/result", 1)]
    stages = []
    for st in _try(race.stages, "stage_url") or []:
        m = re.search(r"stage-(\d+)", st["stage_url"])
        stages.append((st["stage_url"], int(m.group(1)) if m else 0))
    return row, stages


def fetch_edition(store, tables, cal_row, year, refresh_unfinished=False, log=print):
    from procyclingstats import RaceStartlist, Stage
    slug = cal_row["pcs_slug"]
    race, stage_urls = race_pages(store, slug, year, cal_row["kind"], cal_row["gender"], refresh=refresh_unfinished)
    tables.add(dict(races=[race]))
    today = pd.Timestamp.today().normalize()
    for url, number in stage_urls:
        try:
            parsed = parse_stage(store.get(Stage, url), race["race_id"], slug, year, number)
            st = parsed["stages"][0]
            # a cached pre-race page of a stage that has been raced since: download it again
            if refresh_unfinished and not st["finished"] and st["date"] and pd.Timestamp(st["date"]) < today:
                parsed = parse_stage(store.get(Stage, url, refresh=True), race["race_id"], slug, year, number)
            tables.add(parsed)
        except Exception as exc:
            log(f"    {url}: {exc}")
    try:
        sl = store.get(RaceStartlist, f"race/{slug}/{year}/startlist", refresh=refresh_unfinished)
        tables.add(dict(startlists=[dict(race_id=race["race_id"], rider_id=rider_id(r["rider_url"]),
                                         team=r.get("team_name"), rider_number=r.get("rider_number"))
                                    for r in _try(sl.startlist) or []]))
    except Exception as exc:
        log(f"    startlist {slug} {year}: {exc}")
    return race


def fetch_riders(store, tables, rider_ids, log=print):
    from procyclingstats import Rider
    for i, rid in enumerate(rider_ids, 1):
        try:
            tables.add(parse_rider(store.get(Rider, f"rider/{rid}")))
        except Exception as exc:
            log(f"    rider {rid}: {exc}")
        if i % 100 == 0:
            log(f"  {i}/{len(rider_ids)} riders")
            tables.save()
