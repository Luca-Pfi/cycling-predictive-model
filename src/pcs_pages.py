"""Page store shared by the ProCyclingStats scrapers.

Every PCS page is kept as raw HTML in a cache folder, named after its URL
(`rider/tadej-pogacar` -> `rider_tadej-pogacar.html`), so it is downloaded
at most once and can be re-parsed at any time.

Pages get into the cache in two ways:
  * downloaded by the `procyclingstats` package (normal requests, with a delay)
  * imported from pages you saved yourself in a browser ("Save page as...",
    HTML only): `import_saved_pages` reads each file's canonical link to find
    which PCS page it is.

With offline=True nothing is downloaded; only cached pages are used.
"""
import re
import shutil
import time
from pathlib import Path

CANONICAL = re.compile(r'<link rel="canonical" href="https?://(?:www\.)?procyclingstats\.com/([^"?#]+)"')
# sub-pages PCS appends to canonical stage URLs
_SUFFIXES = ("/result/result", "/result", "/gc", "/overview")


def normalise(url):
    url = url.strip().strip("/")
    url = re.sub(r"^https?://(www\.)?procyclingstats\.com/", "", url)
    for s in _SUFFIXES:
        if url.endswith(s) and url.count("/") > 2:
            url = url[: -len(s)]
    return url


def cache_path(cache_dir, url):
    # also safe on Windows for ranking URLs with ?date=...&offset=...
    return Path(cache_dir) / (re.sub(r"[^\w.-]", "_", normalise(url)) + ".html")


class PageStore:
    def __init__(self, cache_dir, offline=False, delay=1.5):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.delay = delay

    def cached(self, prefix=""):
        """Relative URLs of cached pages, e.g. prefix='rider_'."""
        return sorted(p.stem.replace("_", "/") for p in self.cache.glob(f"{prefix}*.html"))

    def get(self, cls, url):
        """A parsed `procyclingstats` object (Rider, Stage, Race, Ranking...) for url."""
        url = normalise(url)
        f = cache_path(self.cache, url)
        if f.exists():
            return cls(url, html=f.read_text(encoding="utf-8"), update_html=False)
        if self.offline:
            raise FileNotFoundError(f"not in cache (offline mode): {url}")
        obj = cls(url)  # normal download by the package
        f.write_text(obj.html.html, encoding="utf-8")
        time.sleep(self.delay)
        return obj


def import_saved_pages(src_dir, cache_dir):
    """Copy browser-saved PCS pages into the cache under their canonical URL."""
    n = 0
    for f in sorted(Path(src_dir).rglob("*")):
        if f.suffix.lower() not in {".html", ".htm", ".txt"} or not f.is_file():
            continue
        m = CANONICAL.search(f.read_text(encoding="utf-8", errors="ignore"))
        if not m:
            print(f"  skipped (no PCS canonical link): {f.name}")
            continue
        dest = cache_path(cache_dir, m.group(1))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        n += 1
    print(f"imported {n} saved pages into {cache_dir}")
    return n
