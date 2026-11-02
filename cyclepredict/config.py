"""Paths and the few definitions every module agrees on."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CALENDAR = DATA / "calendar.csv"
REAL = DATA / "real"          # tables parsed from ProCyclingStats
DEMO = DATA / "demo"          # synthetic seasons in the same schema
PAGES = DATA / "pcs_pages"    # raw HTML cache (not committed)
MODELS = ROOT / "models"
PREDICTIONS = DATA / "predictions"   # every preview / live snapshot, for the track record
LIVE_LOG = DATA / "live_log"

# A rider was "in the breakaway" when PCS records at least this many km ahead of
# the peloton: max(BREAK_MIN_KM, BREAK_MIN_SHARE * stage distance). This separates
# the day's move (usually 50-90% of the race) from late attacks.
BREAK_MIN_KM = 20.0
BREAK_MIN_SHARE = 0.25


def data_dir():
    """Real data when it has been fetched, otherwise the demo seasons."""
    return REAL if (REAL / "stages.csv").exists() else DEMO


def is_demo(path=None):
    return Path(path or data_dir()).resolve() == DEMO.resolve()
