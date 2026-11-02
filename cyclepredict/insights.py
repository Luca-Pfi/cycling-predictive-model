"""Key paths to victory: concrete, data-backed advice per rider.

Built from (1) how this race has been won before (sprint / small group / solo,
how far out winning solos went, how big successful breaks were), (2) each
rider's profile from past results on each terrain, and (3) the chase model
(the gap a break needs at each distance to go).
"""
import numpy as np
import pandas as pd

from cyclepredict.features import ICON_GROUP
from cyclepredict.live import LiveModel, LiveState

TYPE = {"flat": "sprinter", "hills": "puncheur", "mountain": "climber", "itt": "time triallist"}


def rider_profiles(hist, as_of, years=2):
    """Mean result percentile per terrain over the last seasons, and break rate."""
    h = hist[(hist["date"] < as_of) & (hist["date"] >= as_of - pd.Timedelta(days=365 * years))]
    pct = h.pivot_table(index="rider_id", columns="grp", values="pct", aggfunc="mean")
    n = h.groupby("rider_id").size()
    brk = h[h["lab"] == 1].groupby("rider_id")["brk"].mean()
    prof = pct.reindex(columns=list(TYPE)).fillna(0.3)
    best = prof.idxmax(axis=1)
    out = pd.DataFrame({"type": best.map(TYPE), "starts": n.reindex(prof.index).fillna(0),
                        "break_rate": brk.reindex(prof.index).fillna(0)})
    for g in TYPE:
        out[f"pct_{g}"] = prof[g]
    out.loc[out["break_rate"] >= 0.12, "type"] = out["type"] + " · breakaway specialist"
    return out


def race_patterns(stages, slug, as_of, profile_group=None):
    """How past editions of this race (same terrain for stage races) were won."""
    s = stages[(stages["slug"] == slug) & (stages["date"] < as_of) & stages["finished"].astype(bool)]
    if profile_group and s["n_stages"].max() > 1:
        same = s[s["profile_icon"].map(ICON_GROUP) == profile_group]
        s = same if len(same) >= 3 else s
    lab = s[s["labelled"].fillna(False).astype(bool)]
    wt = s["win_type"].value_counts(normalize=True)
    solo = s.loc[s["win_type"] == "solo", "solo_km"].dropna()
    won = lab[lab["break_won"].astype(bool)]
    return dict(
        n=int(len(s)), n_labelled=int(len(lab)),
        break_win_rate=float(lab["break_won"].mean()) if len(lab) else np.nan,
        break_size=float(lab["break_size"].median()) if len(lab) else np.nan,
        winning_break_size=float(won["break_size"].median()) if len(won) else np.nan,
        sprint_share=float(wt.get("bunch sprint", 0)), small_group_share=float(wt.get("small group", 0)),
        solo_share=float(wt.get("solo", 0)), solo_km_median=float(solo.median()) if len(solo) else np.nan,
        solo_km_q=(float(solo.quantile(0.25)), float(solo.quantile(0.75))) if len(solo) >= 3 else None,
    )


def _pct(x):
    return f"{100 * x:.0f}%"


def key_paths(stage, rider, profile, patterns, p_break_wins, live=None, break_riders=None, p_survive=None):
    """Advice for one rider. stage: stage row; rider: prediction row; returns list of strings."""
    icon, grp = stage.get("profile_icon"), ICON_GROUP.get(stage.get("profile_icon"), "hills")
    rtype = profile.get("type", "rider") if profile is not None else "rider"
    route_break = p_break_wins * rider.get("p_break", 0) * rider.get("p_win_if_break", 0)
    route_bunch = (1 - p_break_wins) * rider.get("p_win_if_caught", 0)
    tips = []
    total = route_break + route_bunch
    if total > 0:
        tips.append(f"Most likely route to victory: {'from the breakaway' if route_break > route_bunch else 'from the favourites group'} "
                    f"({_pct(route_break / total)} via the break, {_pct(route_bunch / total)} via the bunch).")
    solo_km, q = patterns.get("solo_km_median"), patterns.get("solo_km_q")
    window = f"{q[0]:.0f}–{q[1]:.0f} km" if q else (f"~{solo_km:.0f} km" if solo_km == solo_km else None)
    last = stage.get("last_climb")
    last_txt = f"{last} (cat. {stage.get('last_climb_cat')})" if isinstance(last, str) else "the last climb"

    in_break = break_riders is not None and rider["rider_id"] in break_riders
    candidate = rider.get("p_break", 0) >= 0.15 and route_break >= route_bunch
    if in_break or (break_riders is None and candidate):
        lm = live or LiveModel()
        need = lm.gap_needed(LiveState(stage_id=stage["stage_id"], km_to_go=50, gap_s=0,
                                       break_riders=list(break_riders or range(int(patterns.get("winning_break_size") or 6))),
                                       hard_finish=icon in ("p3", "p5")))
        if not in_break:
            size = patterns.get("winning_break_size")
            tips.append(f"Breakaway candidate ({_pct(rider['p_break'])} to make the move); the break wins this stage with "
                        f"{_pct(p_break_wins)}." + (f" Breaks that survived here had a median of {size:.0f} riders: "
                                                    f"push for a big group." if size == size else ""))
        tips.append("For the break to stay away it needs about " + ", ".join(
            f"{v // 60}:{v % 60:02d} at {k} km" for k, v in need.items() if k in (50, 30, 20, 10)) + " to go (50/50 line).")
        fastest = rider.get("finale_rank", 99) <= 1
        if fastest:
            tips.append("Likely the fastest finisher of the break: keep working until the gap is safe, then follow "
                        "moves and wait for the sprint.")
        else:
            tips.append("Not the fastest in the break: attack it before the final kilometre"
                        + (f" — winning solos here went {window} from the line." if window else ".")
                        + (f" {last_txt} is the natural springboard." if isinstance(last, str) else ""))
        if p_survive is not None:
            tips.append(f"Live: the break survives with {_pct(p_survive)}; "
                        + ("cooperate, the gap is not safe yet." if p_survive < 0.6 else "the gap looks safe, the race is now inside the break."))
        return tips

    if "sprinter" in rtype and grp == "flat":
        tips.append(f"Needs the bunch sprint ({_pct(patterns.get('sprint_share', 0))} of past editions/stages ended that way): "
                    "his team must keep the break within reach.")
    elif "puncheur" in rtype or (grp == "hills" and icon == "p3"):
        tips.append(f"Wait for the uphill finish / final climb {last_txt}; attack inside the last 2 km."
                    if icon == "p3" else f"Attack on {last_txt} or in the final kilometres before a reduced sprint.")
    elif "climber" in rtype and grp == "mountain":
        tips.append(f"Make the race hard and attack on {last_txt}" + (", ideally in the steepest part of its final kilometres." if icon == "p5" else
                    ", then hold the gap on the descent/run-in."))
    elif "time triallist" in rtype:
        tips.append("Late long-range attack before the sprinters' teams organise, or bridge to the break on a climb.")
    if patterns.get("solo_share", 0) >= 0.3 and window:
        tips.append(f"Solo wins are common here ({_pct(patterns['solo_share'])}): the winning move typically goes {window} from the finish.")
    if p_break_wins >= 0.4:
        tips.append(f"The break is favoured ({_pct(p_break_wins)}): his team needs riders in it or must chase from far out.")
    return tips
