"""Inventory of the data the `procyclingstats` package (PyPI) can extract from PCS.

For every page type it parses example pages with the package's own `parse()`
and writes a catalog: each field, its type, table sizes/columns and an example.

    # offline: real PCS pages saved in the package's test suite
    #   git clone --depth 1 https://github.com/themm1/procyclingstats pcs-src
    python explore_pcs.py --pages pcs-src/tests/fixtures

    # live: download one example page per type (run on your own computer)
    python explore_pcs.py --live

Writes pcs_data_catalog.md (and prints a short summary).
"""
import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import procyclingstats as pcs

CANONICAL = re.compile(r'<link rel="canonical" href="https?://(?:www\.)?procyclingstats\.com/([^"?#]+)"')
LIVE_EXAMPLES = [  # one page per type, used with --live
    "race/tour-de-france/2024",
    "race/tour-de-france/2024/stage-15",
    "race/paris-roubaix/2024/result",
    "race/tour-de-france/2024/startlist",
    "race/tour-de-france/2024/route/climbs",
    "rider/tadej-pogacar",
    "rider/tadej-pogacar/results",
    "team/uae-team-emirates-2024",
    "rankings/me/individual",
    "races.php",
]


def scraper_for(url):
    """(package class, url the class expects) for a PCS page url."""
    u = url.strip("/")
    if u.startswith("rankings"):
        return pcs.Ranking, u
    if u.startswith("team/"):
        return pcs.Team, re.sub(r"/overview.*$", "", u)
    if u.startswith("rider/"):
        return (pcs.RiderResults, u) if "/results" in u else (pcs.Rider, u)
    if u.startswith("race/"):
        if "startlist" in u:
            return pcs.RaceStartlist, re.sub(r"(/startlist)+$", "/startlist", u)
        if "climbs" in u:
            return pcs.RaceClimbs, u
        if "combative" in u:
            return pcs.RaceCombativeRiders, u
        if re.search(r"/stage-\d+|/result", u):
            u = re.sub(r"(/result)+$", "/result", u)
            return pcs.Stage, re.sub(r"(stage-\d+[a-z]?)/result$", r"\1", u)
        return pcs.Race, u
    if u.startswith("races.php") or u == "":
        return pcs.TodayRaces, u or "races.php"
    return None, u


def describe(value):
    """(kind, details, example) for one parsed field."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        cols = list(value[0].keys())
        return "table", f"{len(value)} rows × {len(cols)} columns: {', '.join(cols)}", value[0]
    if isinstance(value, list):
        return "list", f"{len(value)} items", value[:3]
    if isinstance(value, dict):
        return "dict", f"keys: {', '.join(map(str, value.keys()))}", value
    if value is None:
        return "missing", "not on this page", None
    return type(value).__name__, "", value


def short(x, n=160):
    s = repr(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def parse_page(cls, url, html=None):
    obj = cls(url, html=html, update_html=False) if html else cls(url)
    return obj.parse()


def collect_offline(folder):
    pages = []
    for f in sorted(Path(folder).glob("*.txt")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        m = CANONICAL.search(html)
        url = m.group(1) if m else f.stem.replace("_", "/")
        # the fixture for combative riders carries the race's canonical url
        if "comative" in f.stem or "combative" in f.stem:
            url = re.sub(r"(race/[^/]+/\d{4}).*", r"\1/results/combative-riders", url)
        pages.append((url, html))
    return pages


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pages", help="folder of saved PCS pages (.txt/.html)")
    src.add_argument("--live", action="store_true", help="download one example page per type")
    ap.add_argument("--out", default="pcs_data_catalog.md")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    pages = collect_offline(args.pages) if args.pages else [(u, None) for u in LIVE_EXAMPLES]
    catalog = defaultdict(lambda: {"pages": [], "fields": {}})
    for url, html in pages:
        cls, u = scraper_for(url)
        if cls is None:
            print(f"  skip (unknown page type): {url}")
            continue
        try:
            data = parse_page(cls, u, html)
        except Exception as exc:
            print(f"  {cls.__name__:<20} {u}: FAILED {type(exc).__name__}: {exc}"[:200])
            continue
        if not html:
            time.sleep(args.delay)
        entry = catalog[cls.__name__]
        entry["pages"].append(u)
        for field, value in data.items():
            kind, details, example = describe(value)
            old = entry["fields"].get(field)
            if old is None or old[0] == "missing":  # keep the first page that has the field
                entry["fields"][field] = (kind, details, example, u)
        filled = sum(v is not None for v in data.values())
        print(f"  {cls.__name__:<20} {u:<55} {filled}/{len(data)} fields")

    lines = ["# What the `procyclingstats` package can extract from PCS", "",
             f"Generated by `explore_pcs.py` from {len(pages)} real PCS pages. "
             "Each section is one page type (package class); each row is one field "
             "returned by the class's `parse()`.", ""]
    for name in sorted(catalog):
        entry = catalog[name]
        doc = (getattr(pcs, name).__doc__ or "").strip().splitlines()[0]
        lines += [f"## {name}", "", doc, "", f"Example pages: {', '.join(f'`{p}`' for p in entry['pages'])}", "",
                  "| Field | Type | Details | Example |", "|---|---|---|---|"]
        for field, (kind, details, example, _) in sorted(entry["fields"].items()):
            ex = short(example).replace("|", "\\|")
            lines.append(f"| `{field}` | {kind} | {details.replace('|', '/')} | {ex if example is not None else ''} |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    total = sum(len(e["fields"]) for e in catalog.values())
    print(f"\n{len(catalog)} page types, {total} fields -> {args.out}")


if __name__ == "__main__":
    main()
