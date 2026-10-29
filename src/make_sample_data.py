"""Generate a realistic *synthetic* pro-cycling dataset (fictional riders and races).

Riders have hidden abilities (sprint, punch, climb, time trial) and an age curve;
stage results come from how well those abilities fit the stage profile, plus
form, noise and the occasional successful breakaway. The models never see the
hidden abilities: they have to learn them from past results.

    python -m src.make_sample_data --out data/sample
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIRST = ["Lucas", "Mathis", "Jonas", "Pieter", "Marco", "Diego", "Tom", "Felix", "Adam",
         "Hugo", "Iván", "Nils", "Oscar", "Rui", "Sam", "Théo", "Victor", "Wout", "Yannick",
         "Aleks", "Bruno", "Carlos", "Dario", "Emil", "Filippo", "Gianni", "Henrik", "Igor",
         "Jakob", "Kasper", "Lennert", "Mikel", "Noah", "Pablo", "Quentin", "Rasmus", "Stijn"]
LAST = ["Verbeke", "Lindqvist", "Moreau", "Bertolini", "Kowal", "Arrieta", "Van Damme",
        "Hoffmann", "Sørensen", "Dubois", "Ricci", "Novak", "Castellano", "Janssens",
        "Larsen", "Fontaine", "Marchetti", "Oliveira", "Brandt", "Kuipers", "Lombardi",
        "Petit", "Zubiri", "Strand", "Claes", "Garnier", "Mancini", "Vos", "Ekström",
        "Duarte", "Haller", "Rossetti", "Leclercq", "Andersen", "Valverdi", "Meier"]
TEAMS = ["Team Aurora", "Velo Nordic", "Alpine Pro Cycling", "Rouleur Racing",
         "Côte d'Or Cycling", "Iberia Ciclismo", "Polder Pro Team", "Dolomiti Racing",
         "Fjord Cycling", "Atlantic Wind", "Bergamo Bikes", "Ardennes Pro",
         "Pyrenees Elite", "Flanders Flyers", "Baltic Velo", "Helvetia Cycling"]
# stage races: (name, number of stages, terrain mix flat/hilly/mountain/itt)
RACES = [
    ("Grand Tour of the Alps", 21, [0.30, 0.25, 0.35, 0.10]),
    ("Vuelta del Sol", 21, [0.25, 0.30, 0.35, 0.10]),
    ("Giro dei Laghi", 21, [0.30, 0.30, 0.30, 0.10]),
    ("Tour of the Lowlands", 7, [0.60, 0.25, 0.00, 0.15]),
    ("Critérium des Monts", 8, [0.15, 0.30, 0.45, 0.10]),
    ("Volta a Costa", 7, [0.30, 0.40, 0.20, 0.10]),
    ("Ardennes Week", 6, [0.15, 0.70, 0.05, 0.10]),
    ("Tour of the North Sea", 6, [0.70, 0.15, 0.00, 0.15]),
]
ARCHETYPES = {  # mean ability (sprint, punch, climb, tt), share of peloton
    "sprinter":  ([0.85, 0.40, 0.10, 0.30], 0.13),
    "puncheur":  ([0.45, 0.85, 0.45, 0.35], 0.13),
    "climber":   ([0.10, 0.50, 0.85, 0.30], 0.13),
    "gc":        ([0.15, 0.55, 0.80, 0.75], 0.08),
    "rouleur":   ([0.40, 0.35, 0.25, 0.80], 0.10),
    "domestique": ([0.35, 0.35, 0.35, 0.35], 0.43),
}


def make_riders(rng, n):
    names, rows = set(), []
    kinds = list(ARCHETYPES)
    probs = np.array([ARCHETYPES[k][1] for k in kinds])
    for i in range(n):
        while (name := f"{rng.choice(FIRST)} {rng.choice(LAST)}") in names:
            pass
        names.add(name)
        kind = rng.choice(kinds, p=probs / probs.sum())
        ab = np.clip(np.array(ARCHETYPES[kind][0]) + rng.normal(0, 0.10, 4), 0, 1)
        talent = rng.normal(0, 0.08) + (0.08 if kind != "domestique" else 0)
        rows.append(dict(rider_id=f"R{i:03d}", name=name, team=TEAMS[i % len(TEAMS)],
                         birth_year=int(rng.integers(1990, 2005)), archetype=kind,
                         sprint=ab[0], punch=ab[1], climb=ab[2], tt=ab[3], talent=talent))
    return pd.DataFrame(rows)


def make_stage(rng, kind):
    if kind == "flat":
        d, e, ps, up = rng.uniform(150, 225), rng.uniform(400, 1700), rng.uniform(0, 35), rng.random() < 0.05
    elif kind == "hilly":
        d, e, ps, up = rng.uniform(140, 215), rng.uniform(1700, 3300), rng.uniform(40, 160), rng.random() < 0.45
    elif kind == "mountain":
        d, e, ps, up = rng.uniform(120, 210), rng.uniform(3000, 5200), rng.uniform(150, 420), rng.random() < 0.6
    else:  # itt
        d, e, ps, up = rng.uniform(12, 48), rng.uniform(40, 900), rng.uniform(0, 70), rng.random() < 0.12
    return dict(distance_km=round(d, 1), elevation_gain_m=int(e), profile_score=int(ps),
                uphill_finish=int(up), is_itt=int(kind == "itt"), stage_type=kind)


def performance(rng, riders, st, age, form):
    """Higher = better. Ability weights depend continuously on the stage profile."""
    ps, up = st["profile_score"], st["uphill_finish"]
    climb_w = np.clip((ps - 50) / 220, 0, 1.3) * (1 + 0.3 * up)
    punch_w = np.exp(-(((ps - 100) / 70) ** 2)) * (1 + 0.5 * up)
    sprint_w = np.clip(1 - ps / 90, 0, 1) * (1 - 0.8 * up)
    if st["is_itt"]:
        score = 1.0 * riders.tt + 0.4 * climb_w * riders.climb
        noise = 0.06
    else:
        score = sprint_w * riders.sprint + punch_w * riders.punch + climb_w * riders.climb
        score = score / max(sprint_w + punch_w + climb_w, 0.5)
        noise = 0.10 + 0.06 * sprint_w  # bunch sprints are chaotic
    score = score + riders.talent - 0.0035 * (age - 27.5) ** 2 + form
    score = score + rng.normal(0, noise, len(riders))
    # breakaway: on hilly/mountain stages a few random riders sometimes stay away
    if not st["is_itt"] and ps > 40 and rng.random() < 0.18:
        lucky = rng.choice(len(riders), size=3, replace=False)
        score.iloc[lucky] += 0.6
    return score


def generate(seed=7, n_riders=220, years=range(2020, 2026)):
    rng = np.random.default_rng(seed)
    riders = make_riders(rng, n_riders)
    form = pd.Series(0.0, index=riders.index)
    kinds = ["flat", "hilly", "mountain", "itt"]
    stages, results = [], []
    for year in years:
        form = 0.6 * form + rng.normal(0, 0.04, len(riders))  # season-to-season form
        age = year - riders.birth_year
        start = pd.Timestamp(f"{year}-02-15")
        for r_i, (race, n_stages, mix) in enumerate(RACES):
            race_start = start + pd.Timedelta(days=int(r_i * 30 + rng.integers(0, 5)))
            w = 1 + riders.talent.clip(lower=-0.1) * 5  # stronger riders race more
            starters = riders.loc[rng.choice(riders.index, size=150, replace=False, p=w / w.sum())]
            race_kinds = rng.choice(kinds, size=n_stages, p=mix)
            for s in range(n_stages):
                st = make_stage(rng, race_kinds[s])
                sid = f"{year}-{r_i}-{s + 1:02d}"
                stages.append(dict(stage_id=sid, race=race, year=year,
                                   date=(race_start + pd.Timedelta(days=s)).date(), **st))
                perf = performance(rng, starters, st, age[starters.index], form[starters.index])
                ranks = (-perf).rank(method="first").astype(int)
                results += [dict(stage_id=sid, rider_id=rid, rank=rk)
                            for rid, rk in zip(starters.rider_id, ranks)]
    return riders, pd.DataFrame(stages), pd.DataFrame(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    riders, stages, results = generate(args.seed)
    riders[["rider_id", "name", "team", "birth_year"]].to_csv(out / "riders.csv", index=False)
    riders.to_csv(out / "riders_hidden_abilities.csv", index=False)  # for sanity checks only
    stages.to_csv(out / "stages.csv", index=False)
    results.to_csv(out / "results.csv", index=False)
    print(f"{len(riders)} riders, {len(stages)} stages, {len(results)} results -> {out}")
