"""Generate a realistic *synthetic* career dataset (fictional riders).

Every rider gets hidden traits: talent, development speed (early vs late
bloomer), peak age and specialty. Lab tests, junior/U23 results, team level,
injuries and PCS points per season are all noisy consequences of those traits,
with feedback loops (good results -> better team -> more points) and
survivorship (weak riders retire early). The models only see the observable
columns and have to work out the hidden traits.

    python -m src.make_career_data --out data/careers
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.make_sample_data import FIRST, LAST

FIRST_EXTRA = ["Arne", "Bjorn", "Cian", "Dylan", "Elia", "Fabio", "Gorka", "Hamish", "Ion",
               "Joris", "Koen", "Luis", "Magnus", "Nairo", "Olav", "Primo", "Quinn", "Remco",
               "Simone", "Tadeo", "Urko", "Vito", "Warren", "Xabi", "Yves"]
LAST_EXTRA = ["Aerts", "Bakker", "Cortes", "De Smet", "Engels", "Ferrand", "Gallo", "Hansen",
              "Iturria", "Jensen", "Kragh", "Lopez", "Maes", "Nilsen", "Orsini", "Peeters",
              "Quintero", "Rinaldi", "Schmid", "Tanaka", "Urban", "Vidal", "Wouters", "Zanetti"]
# nationality: (share of riders, development-system strength)
NATIONS = {"BEL": (0.13, 0.35), "FRA": (0.12, 0.25), "ITA": (0.11, 0.25), "ESP": (0.09, 0.20),
           "NED": (0.08, 0.35), "GBR": (0.05, 0.30), "DEN": (0.05, 0.35), "GER": (0.05, 0.15),
           "SLO": (0.03, 0.30), "COL": (0.05, 0.05), "USA": (0.04, 0.05), "AUS": (0.05, 0.15),
           "NOR": (0.03, 0.25), "SUI": (0.03, 0.15), "OTHER": (0.09, -0.10)}
# specialty: share, peak-age offset, weight kg, ftp bonus, sprint bonus (W)
SPECIALTIES = {
    "sprinter":  (0.17, -0.8, 75, -0.20, 330),
    "classics":  (0.17, 0.3, 74, -0.05, 150),
    "puncheur":  (0.16, -0.2, 66, 0.10, 90),
    "climber":   (0.18, 0.4, 61, 0.30, -60),
    "gc":        (0.10, 1.0, 64, 0.35, -30),
    "rouleur":   (0.22, 0.6, 72, 0.00, 40),
}
BACKGROUNDS = {"road": (0.70, 0.0, 0.0), "mtb": (0.10, 0.15, -0.3),
               "cyclocross": (0.10, 0.15, -0.3), "track": (0.07, 0.10, -0.1),
               "triathlon": (0.03, -0.05, 0.6)}  # share, talent boost, dev-speed shift
LEVEL_BONUS = {"amateur": -1.0, "conti": -0.35, "pro": 0.15, "wt": 0.55}
FIRST_SEASON, LAST_SEASON, MAX_AGE = 1995, 2025, 38


def _pick(rng, table, n):
    keys = list(table)
    p = np.array([v[0] for v in table.values()])
    return rng.choice(keys, size=n, p=p / p.sum())


def make_riders(rng, n):
    firsts, lasts = FIRST + FIRST_EXTRA, LAST + LAST_EXTRA
    combos = rng.choice(len(firsts) * len(lasts), size=n, replace=False)
    names = [f"{firsts[c // len(lasts)]} {lasts[c % len(lasts)]}" for c in combos]

    nat = _pick(rng, NATIONS, n)
    spec = _pick(rng, SPECIALTIES, n)
    bg = _pick(rng, BACKGROUNDS, n)
    nat_strength = np.array([NATIONS[x][1] for x in nat])
    sp = np.array([SPECIALTIES[x][1:] for x in spec])
    bg_v = np.array([BACKGROUNDS[x][1:] for x in bg])

    talent = rng.normal(0, 1, n) + bg_v[:, 0]
    dev_speed = rng.normal(0, 1, n) + bg_v[:, 1]  # >0 = early bloomer
    started_age = np.clip(np.round(rng.normal(13 - 0.8 * dev_speed, 2.5)), 6, 22)
    late_start = np.clip(started_age - 14, 0, None)
    dev_speed = dev_speed - 0.12 * late_start  # late starters develop later
    peak_age = 27.6 + sp[:, 0] - 1.1 * dev_speed + rng.normal(0, 0.9, n)

    weight = sp[:, 1] + rng.normal(0, 3.5, n)
    height = 150 + 0.42 * weight + rng.normal(0, 4.5, n)
    # lab tests at 19: noisy measurements of talent, shaped by specialty
    vo2 = 74 + 3.2 * talent + 1.2 * dev_speed + 8 * sp[:, 2] + rng.normal(0, 2.8, n)
    ftp = 5.0 + 0.28 * talent + 0.12 * dev_speed + sp[:, 2] + rng.normal(0, 0.22, n)
    sprint_w = 1150 + sp[:, 3] + 6 * (weight - 68) + 45 * talent + rng.normal(0, 90, n)

    junior_perf = talent + 0.8 * dev_speed + nat_strength + rng.normal(0, 0.7, n)
    junior_worlds_top10 = (junior_perf > 2.3).astype(int)
    junior_nc_points = np.round(np.clip(np.exp(1.0 * junior_perf + 2.0) - 8, 0, None))
    devo_team = (rng.random(n) < 1 / (1 + np.exp(-(1.4 * junior_perf + 2 * nat_strength - 2.2)))).astype(int)

    return pd.DataFrame(dict(
        rider_id=[f"C{i:04d}" for i in range(n)], name=names, nationality=nat,
        birth_year=rng.integers(FIRST_SEASON - 19, LAST_SEASON - 18, n),
        specialty=spec, background=bg, started_cycling_age=started_age.astype(int),
        height_cm=height.round(0), weight_kg=weight.round(1),
        vo2max=vo2.round(1), ftp_wkg_u23=ftp.round(2), sprint_peak_w=sprint_w.round(0),
        junior_worlds_top10=junior_worlds_top10, junior_nc_points=junior_nc_points,
        devo_team=devo_team,
        # hidden truth (saved separately, never used for training)
        h_talent=talent, h_dev_speed=dev_speed, h_peak_age=peak_age, h_nat=nat_strength,
    ))


def simulate_careers(rng, riders):
    rows = []
    for r in riders.itertuples():
        level = "conti" if r.devo_team else "amateur"
        retired = False
        form = 0.0
        prev_lp = None
        for age in range(19, MAX_AGE + 1):
            season = r.birth_year + age
            if season > LAST_SEASON or retired:
                break
            # age curve: rise to the peak (slower for late bloomers), slow decline
            gap = age - r.h_peak_age
            rise = 0.045 * (1 - 0.25 * r.h_dev_speed)
            shape = -rise * gap**2 if gap < 0 else -0.03 * gap**2
            form = 0.5 * form + rng.normal(0, 0.25)  # good and bad years carry over
            injured = rng.random() < 0.06 + 0.004 * max(age - 28, 0)
            lp = (3.9 + 1.45 * r.h_talent + shape + LEVEL_BONUS[level] + 0.4 * r.h_nat * (age < 23)
                  + form - 1.0 * injured + rng.normal(0, 0.35))
            lp = 9.0 - np.logaddexp(0, 9.0 - lp)  # soft ceiling: nobody scores 20,000 points
            points = max(0.0, round(float(np.expm1(lp)) if lp > 0 else 0.0))
            lp_obs = np.log1p(points)
            race_days = int(np.clip(rng.normal({"amateur": 35, "conti": 50, "pro": 62, "wt": 68}[level]
                                               - 22 * injured, 7), 3, 95))
            wins = int(rng.poisson(min(np.exp(1.05 * (lp_obs - 6.1)), 18)))
            top10s = int(min(rng.poisson(np.exp(0.75 * (lp_obs - 3.6)) * race_days / 60), 0.6 * race_days))
            gt_starts = int(level == "wt") * int(rng.binomial(2, min(0.9, 0.15 + 0.12 * max(lp_obs - 4, 0))))
            u23 = age <= 22 and level != "wt"
            u23_wins = int(rng.poisson(min(np.exp(1.1 * (lp - 3.6)), 7))) if u23 else 0
            u23_podiums = u23_wins + int(rng.poisson(min(np.exp(0.9 * (lp - 3.1)), 9))) if u23 else 0
            avenir = np.nan
            if u23 and age >= 20 and rng.random() < 0.5 + 0.2 * r.devo_team:
                rank = int(np.clip(np.round(60 - 14 * lp + rng.normal(0, 8)), 1, 140))
                avenir = rank if rank <= 30 else np.nan
            u23_worlds = np.nan
            if u23 and rng.random() < 0.45:
                rank = int(np.clip(np.round(55 - 12 * lp + rng.normal(0, 10)), 1, 150))
                u23_worlds = rank if rank <= 40 else np.nan
            rows.append(dict(rider_id=r.rider_id, season=season, age=age, team_level=level,
                             race_days=race_days, points=points, wins=wins, top10s=top10s,
                             gt_starts=gt_starts, injured=int(injured), u23_wins=u23_wins,
                             u23_podiums=u23_podiums, avenir_rank=avenir, u23_worlds_rank=u23_worlds))
            # next season: team level follows results (with some stickiness)
            trend = lp_obs if prev_lp is None else 0.6 * lp_obs + 0.4 * prev_lp
            prev_lp = lp_obs
            target = "wt" if trend > 5.6 else "pro" if trend > 4.3 else "conti" if trend > 2.6 else "amateur"
            order = list(LEVEL_BONUS)
            step = np.sign(order.index(target) - order.index(level))
            if step and rng.random() < 0.75:
                level = order[order.index(level) + int(step)]
            # retirement / dropping out
            hazard = 1 / (1 + np.exp(-(-4.6 + 0.33 * (age - 28) - 0.9 * (lp_obs - 4.0)
                                       + 1.2 * (level == "amateur" and age >= 23))))
            retired = rng.random() < hazard
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/careers")
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    riders = make_riders(rng, args.n)
    seasons = simulate_careers(rng, riders)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = [c for c in riders if c.startswith("h_")]
    riders.drop(columns=hidden).to_csv(out / "riders.csv", index=False)
    riders[["rider_id", *hidden]].to_csv(out / "riders_hidden_traits.csv", index=False)
    seasons.to_csv(out / "seasons.csv", index=False)
    peak = seasons.groupby("rider_id")["points"].max()
    print(f"{len(riders)} riders, {len(seasons)} rider-seasons -> {out}")
    print("peak-season points quantiles:", peak.quantile([.5, .9, .99, 1]).round().to_dict())
    print("career length (seasons):", seasons.groupby("rider_id").size().describe().round(1).to_dict())
