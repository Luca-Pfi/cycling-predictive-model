"""Live breakaway odds from the race situation.

State: km to go, gap of the break (s), who is in it, how hard the peloton chases.
The chase model: a peloton chasing with intent closes about 1 minute per 10 km
(6 s/km), more in the last 10 km and less when only one team rides; a bigger,
stronger break loses less. The break survives if its gap is larger than what the
peloton can close before the line (+ a margin for the finale), with uncertainty.

Until enough live snapshots with outcomes have been logged (data/live_log), these
rule-of-thumb coefficients are the model; `LiveModel.fit` then learns them from
the logged races. The pre-race probability is blended in early in the race.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cyclepredict import config

CHASE = {"none": 2.0, "one team": 4.5, "several teams": 6.5, "all out": 8.5}  # s per km the peloton gains


@dataclass
class LiveState:
    stage_id: str
    km_to_go: float
    gap_s: float
    break_riders: list = field(default_factory=list)
    chase: str = "several teams"
    hard_finish: bool = False   # summit / steep finish still to come
    timestamp: str = ""


class LiveModel:
    def __init__(self, sd=2.2, finale_margin=12.0, size_effect=0.03):
        self.sd, self.finale_margin, self.size_effect = sd, finale_margin, size_effect
        self.fitted = None  # logistic model learned from logged snapshots

    def closing_rate(self, s: LiveState):
        c = CHASE.get(s.chase, 6.5)
        c *= 1 - self.size_effect * (max(len(s.break_riders), 1) - 1)    # bigger breaks hold out better
        c *= 1.25 if s.km_to_go <= 10 else 1.0                             # the finale is faster
        c *= 1.2 if s.hard_finish else 1.0                                 # climbs favour the strongest
        return max(c, 0.5)

    def p_survive_physics(self, s: LiveState):
        if s.km_to_go <= 0:
            return float(s.gap_s > 0)
        need = self.closing_rate(s) * s.km_to_go + self.finale_margin
        # uncertainty grows with the distance left
        sd = self.sd * s.km_to_go ** 0.75 + 5
        return float(norm.cdf((s.gap_s - need) / sd))

    def p_survive(self, s: LiveState, p_pre=None, distance_km=None):
        if self.fitted is not None:
            p = float(self.fitted.predict_proba(self._x([s]))[0, 1])
        else:
            p = self.p_survive_physics(s)
        if p_pre is None or not distance_km:
            return p
        # early in the race the pre-race model knows more than the gap does
        w = np.clip(s.km_to_go / distance_km, 0, 1)  # weight of the pre-race prior
        lo = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
        return float(1 / (1 + np.exp(-((1 - w) * lo(p) + w * lo(p_pre)))))

    def gap_needed(self, s: LiveState, target=0.5, km_points=(80, 50, 30, 20, 10, 5)):
        """Gap (s) the break needs at each distance to go for P(survive) = target."""
        out = {}
        for km in km_points:
            probe = LiveState(**{**asdict(s), "km_to_go": km})
            lo, hi = 0.0, 1800.0
            for _ in range(40):
                mid = (lo + hi) / 2
                probe.gap_s = mid
                lo, hi = (lo, mid) if self.p_survive_physics(probe) >= target else (mid, hi)
            out[km] = round(hi)
        return out

    @staticmethod
    def _x(states):
        return np.array([[np.log1p(s.gap_s), np.log1p(s.km_to_go), len(s.break_riders),
                          CHASE.get(s.chase, 6.5), float(s.hard_finish)] for s in states])

    def fit(self, stages, log_dir=config.LIVE_LOG, min_races=30):
        """Learn the live model from logged snapshots of stages that have finished."""
        from sklearn.linear_model import LogisticRegression
        snaps = load_log(log_dir, stages)
        snaps = snaps.dropna(subset=["break_won"]) if "break_won" in snaps else snaps
        if snaps.empty or "break_won" not in snaps or snaps["stage_id"].nunique() < min_races:
            return False
        states = [LiveState(**{k: r[k] for k in LiveState.__dataclass_fields__}) for _, r in snaps.iterrows()]
        self.fitted = LogisticRegression(C=1.0).fit(self._x(states), snaps["break_won"].astype(int))
        return True


def rider_odds(predictor, rows, stage_row, state: LiveState, p_surv):
    """Live P(win) and fair odds per rider, given who is in the break."""
    from cyclepredict.models import WIN_FEATURES
    r = rows[rows["stage_id"] == state.stage_id].copy()
    inb = r["rider_id"].isin(state.break_riders).to_numpy()
    fo = predictor.finale.predict_proba(r[WIN_FEATURES])[:, 1]
    fo = fo / (1 - fo + 1e-9)
    q = np.where(inb, fo, 0)
    q = q / q.sum() if q.sum() > 0 else q
    fav = predictor.favourite.predict_proba(r[WIN_FEATURES])[:, 1] * ~inb
    fav = fav / fav.sum() if fav.sum() > 0 else fav
    r["in_break"] = inb
    r["p_win_if_survives"] = q
    r["p_win"] = p_surv * q + (1 - p_surv) * fav
    r["fair_odds"] = 1 / r["p_win"].clip(lower=1e-4)
    return r.sort_values("p_win", ascending=False)


def log_snapshot(state: LiveState, p_surv, log_dir=config.LIVE_LOG):
    """Keep every live estimate: once the stage is over it becomes training data."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {**asdict(state), "p_survive": p_surv, "timestamp": state.timestamp or pd.Timestamp.now().isoformat()}
    with open(log_dir / "snapshots.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_log(log_dir=config.LIVE_LOG, stages=None):
    f = Path(log_dir) / "snapshots.jsonl"
    if not f.exists():
        return pd.DataFrame()
    snaps = pd.DataFrame([json.loads(line) for line in f.read_text().splitlines() if line.strip()])
    if stages is not None and len(snaps):
        snaps = snaps.merge(stages[["stage_id", "break_won"]].dropna(), on="stage_id", how="left")
    return snaps
