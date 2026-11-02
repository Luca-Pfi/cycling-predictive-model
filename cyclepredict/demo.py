"""Synthetic seasons for the calendar's races, in exactly the PCS table schema.

Only for building and testing while real PCS data is not available: riders and
teams are fictional, races follow the real calendar. Hidden rider traits (sprint,
punch, climbing, cobbles, time trial, engine, aggressiveness, talent) and a hidden
"break friendliness" per race drive breakaways, chases, GC and results.

    python -m cyclepredict demo
"""
import numpy as np
import pandas as pd

from cyclepredict import config

DEMO_TODAY = pd.Timestamp("2026-07-12")  # the demo's clock: mid Tour de France 2026
YEARS = range(2020, 2027)
GRAND_TOURS = {"tour-de-france", "giro-d-italia", "vuelta-a-espana"}
UWT = GRAND_TOURS | {"liege-bastogne-liege", "milano-sanremo", "paris-roubaix", "ronde-van-vlaanderen",
                     "amstel-gold-race", "la-fleche-wallone", "il-lombardia", "dauphine", "san-sebastian",
                     "tirreno-adriatico", "tour-down-under", "gent-wevelgem", "strade-bianche",
                     "omloop-het-nieuwsblad", "bretagne-classic", "cyclassics-hamburg", "e3-harelbeke",
                     "benelux-tour", "itzulia-basque-country", "volta-a-catalunya", "gp-montreal",
                     "gp-quebec", "tour-de-suisse", "tour-de-romandie", "paris-nice", "uae-tour",
                     "classic-brugge-de-panne", "dwars-door-vlaanderen", "eschborn-frankfurt",
                     "tour-de-pologne", "cadel-evans-great-ocean-road-race", "copenhagen-sprint",
                     "tour-of-guangxi", "tour-de-france-femmes"}
STAGES = {"tour-down-under": 6, "paris-nice": 8, "tirreno-adriatico": 7, "volta-a-catalunya": 7,
          "itzulia-basque-country": 6, "tour-de-romandie": 6, "dauphine": 8, "tour-de-suisse": 8,
          "tour-de-pologne": 7, "uae-tour": 7, "tour-de-france-femmes": 8}
MIX = {  # race terrain -> share of flat / hilly / mountain / itt stages
    "mountain": [0.30, 0.25, 0.35, 0.10], "mixed": [0.40, 0.30, 0.20, 0.10],
    "hilly": [0.35, 0.45, 0.10, 0.10], "flat": [0.60, 0.25, 0.00, 0.15], "cobbles": [0.5, 0.4, 0, 0.1]}
STAGE_POINTS = [100, 80, 65, 55, 45, 35, 30, 25, 20, 17, 15, 14, 13, 12, 11]
NATIONS = ["BE", "FR", "IT", "ES", "NL", "DK", "GB", "DE", "SI", "CO", "US", "AU", "NO", "CH", "PT", "PL", "AT"]
FIRST = ["Arno", "Bram", "Cas", "Dorian", "Eli", "Fenn", "Gino", "Hugo", "Ivo", "Jory", "Kai", "Lenz",
         "Milo", "Nico", "Oren", "Pim", "Quin", "Remi", "Sten", "Timo", "Ugo", "Vik", "Wes", "Yan", "Zeno",
         "Alba", "Bea", "Cleo", "Dana", "Elin", "Fay", "Gaia", "Hana", "Iris", "Jade", "Kira", "Lise"]
SYL = ["van", "ber", "dor", "mel", "kas", "tor", "rin", "vel", "sor", "lan", "mar", "dek", "fol",
       "gar", "hen", "jor", "lus", "nov", "pel", "quin", "ros", "sel", "tam", "vor", "wik", "zan"]


def _name(rng, women):
    first = rng.choice(FIRST[25:] if women else FIRST[:25])
    last = "".join(rng.choice(SYL, size=rng.integers(2, 4))).capitalize()
    return f"{first} {last}"


def make_riders(rng, n, women, n_wt, n_pro):
    g = "w" if women else "m"
    teams = [(f"{'Femmes ' if women else ''}{t} {'Pro' if i >= n_wt else 'Racing'}", "wt" if i < n_wt else "pro")
             for i, t in enumerate(["Aurora", "Nordlys", "Altitude", "Polder", "Ardennes", "Iberia", "Dolomia",
                                    "Fjord", "Atlantis", "Borealis", "Cantabria", "Helvetia", "Baltica",
                                    "Lombarda", "Provence", "Kempen", "Vosges", "Andes", "Danube", "Riviera",
                                    "Calypso", "Meseta", "Tundra", "Savoia", "Zephyr", "Galicia", "Orkney",
                                    "Valais", "Brabant", "Etna"][: n_wt + n_pro])]
    kind = rng.choice(["sprinter", "puncheur", "climber", "rouleur", "cobbles", "gc", "domestique"],
                      p=[0.12, 0.13, 0.15, 0.15, 0.08, 0.07, 0.30], size=n)
    base = {"sprinter": [.9, .4, .1, .3, .4, .5], "puncheur": [.4, .9, .5, .4, .4, .5],
            "climber": [.1, .5, .9, .3, .1, .4], "rouleur": [.3, .3, .3, .7, .5, .9],
            "cobbles": [.5, .5, .1, .5, .9, .8], "gc": [.1, .6, .9, .8, .2, .5],
            "domestique": [.35, .35, .35, .35, .35, .6]}
    ab = np.clip(np.array([base[k] for k in kind]) + rng.normal(0, 0.12, (n, 6)), 0, 1)
    talent = rng.normal(0, 1, n) + np.where(kind == "domestique", -0.6, 0.25)
    team_idx = np.concatenate([np.repeat(np.arange(len(teams)), n // len(teams)),
                               rng.integers(0, len(teams), n - n // len(teams) * len(teams))])
    rng.shuffle(team_idx)
    nat = rng.choice(NATIONS, size=n, p=np.r_[[.12, .12, .11, .1, .08], np.full(12, .47 / 12)])
    return pd.DataFrame(dict(
        rider_id=[f"{g}{i:04d}-" + _name(rng, women).lower().replace(" ", "-") for i in range(n)],
        gender="women" if women else "men", kind=kind,
        sprint=ab[:, 0], punch=ab[:, 1], climb=ab[:, 2], tt=ab[:, 3], cobbles=ab[:, 4], engine=ab[:, 5],
        aggression=rng.normal(0, 1, n) + np.where(np.isin(kind, ["rouleur", "domestique"]), 0.5, 0),
        talent=talent, team=[teams[t][0] for t in team_idx], team_class=[teams[t][1] for t in team_idx],
        nationality=nat, birth_year=rng.integers(1988, 2005, n),
        height=np.round(rng.normal(1.80, 0.06, n) - np.where(kind == "climber", .06, 0), 2),
        weight=np.round(rng.normal(70, 6, n) - np.where(kind == "climber", 7, 0) + np.where(kind == "sprinter", 5, 0)),
    ))


def stage_profile(rng, kind, one_day, gt):
    """kind -> profile fields, like a PCS stage page before the race."""
    if kind == "itt":
        d, v, ps, icon = rng.uniform(12, 45), rng.uniform(50, 600), rng.uniform(0, 60), "p1"
    elif kind == "flat":
        d, v, ps = rng.uniform(150, 215), rng.uniform(300, 1500), rng.uniform(0, 40)
        icon = "p1"
    elif kind == "cobbles":
        d, v, ps = rng.uniform(195, 265), rng.uniform(1200, 2600), rng.uniform(30, 95)
        icon = rng.choice(["p1", "p2"])
    elif kind == "hilly":
        d = rng.uniform(200, 265) if one_day else rng.uniform(150, 215)
        v, ps = rng.uniform(1800, 3800), rng.uniform(50, 175)
        icon = rng.choice(["p2", "p3"], p=[0.6, 0.4])
    else:
        d, v, ps = rng.uniform(130, 245 if one_day else 205), rng.uniform(3000, 5200), rng.uniform(150, 420)
        icon = rng.choice(["p4", "p5"], p=[0.4, 0.6])
    cats = {"flat": [0.1, 0.2], "cobbles": [0, 0.3], "hilly": [0.6, 2.5], "mountain": [2.2, 1.5], "itt": [0, 0]}[kind]
    n_hard, n_easy = min(rng.poisson(cats[0]) + (kind == "mountain"), 5), rng.poisson(cats[1])
    climbs = list(rng.choice(["HC", "1", "2"], n_hard, p=[.25, .5, .25]) if kind == "mountain" else rng.choice(["2", "3"], n_hard)) \
        + list(rng.choice(["3", "4"], n_easy))
    rng.shuffle(climbs)
    if kind == "mountain" and climbs:  # the hardest climb comes last on mountain stages
        climbs.sort(key=lambda c: {"HC": 5, "1": 4, "2": 3, "3": 2, "4": 1}[c])
    return dict(distance_km=round(d, 1), vertical_m=int(v), profile_score=int(ps), profile_icon=icon,
                stage_type="ITT" if kind == "itt" else "RR", kind=kind, climbs=climbs)


def finish_ability(r, kind, icon):
    if kind == "itt":
        return r["tt"]
    if kind == "flat":
        return r["sprint"]
    if kind == "cobbles":
        return 0.6 * r["cobbles"] + 0.4 * r["sprint"]
    if kind == "hilly":
        return r["punch"] if icon == "p3" else 0.55 * r["punch"] + 0.45 * r["sprint"]
    return r["climb"]


def simulate_stage(rng, st, sl, race, ctx):
    """One stage: breakaway, chase, finale. Returns results, gc deltas, climbs, won_how."""
    n = len(sl)
    kind, icon, dist = st["kind"], st["profile_icon"], st["distance_km"]
    fin = finish_ability(sl, kind, icon).to_numpy()
    strength = sl["talent"].to_numpy() * 0.35 + fin + rng.normal(0, 0.12 + 0.15 * (kind == "flat"), n)
    breakaway = np.zeros(n)
    in_break = np.zeros(n, bool)
    survived, won_how, solo_km = False, None, 0.0
    if kind != "itt":
        size = {"flat": 3, "cobbles": 7, "hilly": 8, "mountain": 16}[kind] * (1.4 if ctx["gt"] else 1.0)
        size = int(np.clip(rng.poisson(size), 1, 35))
        terrain = {"flat": 0, "cobbles": sl["cobbles"], "hilly": sl["punch"], "mountain": sl["climb"]}[kind]
        gc_threat = ctx["gc_rank"].reindex(sl["rider_id"]).fillna(999).to_numpy() <= 20
        w = (1.4 * sl["aggression"] + 0.8 * sl["engine"] + 0.7 * (sl["team_class"] == "pro") +
             0.7 * terrain - 1.6 * (sl["talent"] > 1.3) - 1.4 * gc_threat - 0.6 * sl["rider_id"].isin(ctx["prev_break"])
             + 0.4 * (sl["nationality"] == race["country"])).to_numpy()
        keys = w + rng.gumbel(size=n)
        members = np.argsort(-keys)[:size]
        in_break[members] = True
        quality = (sl["engine"].to_numpy()[members].mean() + np.mean(fin[members])) - (fin.mean() + 0.5)
        logit = ({"flat": -1.9, "cobbles": -0.9, "hilly": 0.0, "mountain": 0.7}[kind] + race["break_bias"]
                 - 0.45 * (icon in ("p3", "p5")) + 0.6 * ctx["progress"] * (not ctx["one_day"])
                 + 0.4 * (ctx["gt"] and kind in ("hilly", "mountain")) + 1.2 * quality
                 - 0.35 * ctx["quality_z"] - 0.12 * ctx["sprint_teams"] * (kind == "flat")
                 - 2.5 * (ctx["last"] and kind == "flat" and ctx["gt"]) - 0.3 * (race["kind"] == "championship")
                 + rng.normal(0, 0.35))
        survived = rng.random() < 1 / (1 + np.exp(-logit))
        km = np.where(in_break, dist * rng.uniform(0.35, 0.95, n), 0)
        breakaway = km
    if survived:
        score = np.where(in_break, 10 + fin + 0.5 * sl["engine"].to_numpy() + rng.normal(0, 0.3, n), strength)
        order = np.argsort(-score)
        breakaway[order[0]] = dist * rng.uniform(0.9, 0.98)
        gap = score[order[0]] - score[order[1]]
        if gap > 0.45 and in_break.sum() > 1:
            solo_km = float(rng.uniform(2, 25))
            won_how = f"{solo_km:.0f} km solo"
        else:
            won_how = "Sprint of small group" if in_break.sum() > 2 else "Sprint of two"
    else:
        score = strength - 0.25 * in_break
        order = np.argsort(-score)
        gap = score[order[0]] - score[order[1]]
        if kind == "itt":
            won_how = "Time trial"
        elif kind == "flat":
            won_how = "Sprint of large group"
        elif kind == "mountain" or (gap > 0.25 and rng.random() < 0.6):
            solo_km = float(rng.uniform(1, 12 if kind == "mountain" else 25))
            won_how = f"{solo_km:.0f} km solo"
            breakaway[order[0]] = solo_km
        else:
            won_how = "Sprint of small group"
    rank = np.empty(n, int)
    rank[order] = np.arange(1, n + 1)
    status = np.where(rng.random(n) < 0.02 * (not ctx["one_day"]) + 0.12 * ctx["one_day"], "DNF", "DF")
    status[order[:3]] = "DF"
    # time lost on GC
    spread = {"flat": 0, "cobbles": 30, "hilly": 40, "mountain": 420, "itt": 6 * dist}[kind]
    gc_s = np.clip((score[order[0]] - score), 0, 3) * spread / 3
    if kind == "flat":
        gc_s = np.where(rank > 150, 20, 0)
    if survived:
        gc_s = np.where(in_break, 0, gc_s + rng.uniform(60, 600))
    climbs = []
    for i, cat in enumerate(st["climbs"]):
        last = i == len(st["climbs"]) - 1
        pool = np.where(in_break)[0] if (in_break.any() and (survived or not last)) else order[:5]
        who = pool[np.argmax(sl["climb"].to_numpy()[pool] + rng.normal(0, 0.2, len(pool)))]
        climbs.append(dict(order=i, climb=f"Côte {i + 1}", category=cat, first_rider=sl["rider_id"].iloc[who]))
    return dict(rank=rank, status=status, breakaway=np.round(breakaway), in_break=in_break,
                survived=survived, won_how=won_how, gc_s=gc_s, climbs=climbs)


def pick_startlist(rng, riders, race, teams_in, per_team, kind_mix):
    pool = riders[riders["team"].isin(teams_in)]
    fit = (kind_mix[0] * pool["sprint"] + kind_mix[1] * pool["punch"] + kind_mix[2] * pool["climb"]
           + 0.3 * pool["cobbles"] * (race["terrain"] == "cobbles") + 0.4 * pool["talent"] + rng.normal(0, 0.3, len(pool)))
    pool = pool.assign(fit=fit).sort_values("fit", ascending=False)
    return pool.groupby("team").head(per_team).reset_index(drop=True)


def generate(seed=3):
    rng = np.random.default_rng(seed)
    cal = pd.read_csv(config.CALENDAR)
    men = make_riders(rng, 900, False, 18, 12)
    women = make_riders(rng, 330, True, 14, 0)
    riders = pd.concat([men, women], ignore_index=True)
    cal = cal.assign(break_bias=rng.normal(0, 0.55, len(cal)), day=rng.integers(1, 22, len(cal)))
    tables = {k: [] for k in ["races", "stages", "results", "gc", "climbs", "startlists"]}
    for year in YEARS:
        for _, race in cal.sort_values(["month", "day"]).iterrows():
            if race["pcs_slug"].startswith("olympic") and year % 4:
                continue
            pool_r = riders[riders["gender"] == race["gender"]]
            race_id = f"{race['pcs_slug']}/{year}"
            start = pd.Timestamp(year=year, month=int(race["month"]), day=int(race["day"]))
            gt = race["pcs_slug"] in GRAND_TOURS
            n_st = 21 if gt else STAGES.get(race["pcs_slug"], 5) if race["kind"] == "stage_race" else 1
            one_day = n_st == 1
            teams = pool_r.groupby("team")["team_class"].first()
            wt, pro = list(teams[teams == "wt"].index), list(teams[teams == "pro"].index)
            if race["kind"].startswith("championship"):
                sl = pool_r.sort_values("talent", ascending=False).groupby("nationality").head(
                    1 if race["kind"] == "championship_itt" else 7).head(60 if race["kind"] == "championship_itt" else 175)
                sl = sl.assign(team=sl["nationality"])
            else:
                big = race["pcs_slug"] in UWT
                invited = wt + list(rng.choice(pro, size=min(len(pro), 4 if big else 8), replace=False)) if big \
                    else list(rng.choice(wt, 9, replace=False)) + pro
                mix = {"flat": [1, .3, 0], "cobbles": [.6, .5, 0], "hilly": [.4, 1, .4], "mountain": [.1, .5, 1],
                       "mixed": [.6, .6, .6]}[race["terrain"]]
                sl = pick_startlist(rng, pool_r, race, invited, 8 if gt else 7, mix)
            end = start + pd.Timedelta(days=n_st - 1 + (2 if gt else 0))
            tables["races"].append(dict(race_id=race_id, slug=race["pcs_slug"], year=year, name=race["name"],
                                        gender=race["gender"], kind=race["kind"],
                                        uci_tour=("2.UWT" if not one_day else "1.UWT") if race["pcs_slug"] in UWT else ("2.Pro" if not one_day else "1.Pro"),
                                        startdate=start.date().isoformat(), enddate=end.date().isoformat()))
            if start - DEMO_TODAY <= pd.Timedelta(days=21):  # startlists appear ~3 weeks before
                tables["startlists"] += [dict(race_id=race_id, rider_id=r.rider_id, team=r.team, rider_number=i + 1)
                                         for i, r in enumerate(sl.itertuples())]
            quality = float(np.exp(1.1 * sl["talent"].clip(-1, 3)).nlargest(40).sum())
            q_z = (quality - 120) / 60
            sprint_teams = int((sl.groupby("team")["sprint"].max() > 0.85).sum())
            mix = MIX[race["terrain"]]
            gc_time = pd.Series(0.0, index=sl["rider_id"])
            prev_break = set()
            for k in range(1, n_st + 1):
                if race["kind"] == "championship_itt":
                    kind = "itt"
                elif one_day or race["kind"] == "championship":
                    kind = {"mixed": "hilly"}.get(race["terrain"], race["terrain"])
                else:
                    kind = rng.choice(["flat", "hilly", "mountain", "itt"], p=mix)
                    if gt and k == n_st:
                        kind = "flat"
                st = stage_profile(rng, kind, one_day, gt)
                date = start + pd.Timedelta(days=k - 1 + (1 if gt and k > 9 else 0) + (1 if gt and k > 15 else 0))
                stage_id = f"race/{race_id}/result" if one_day else f"race/{race_id}/stage-{k}"
                row = dict(stage_id=stage_id, race_id=race_id, slug=race["pcs_slug"], year=year, stage_number=k,
                           date=date.date().isoformat(), **{c: st[c] for c in ["distance_km", "vertical_m", "profile_score",
                                                                              "profile_icon", "stage_type"]},
                           departure=None, arrival=None, won_how=None, startlist_quality=round(quality * 6),
                           n_climbs=len(st["climbs"]), **{f"climbs_{c.lower()}": st["climbs"].count(c) for c in ["HC", "1", "2", "3", "4"]},
                           last_climb=f"Côte {len(st['climbs'])}" if st["climbs"] else None,
                           last_climb_cat=st["climbs"][-1] if st["climbs"] else None, finished=False)
                if date >= DEMO_TODAY:
                    tables["stages"].append(row)
                    continue
                gc_rank = gc_time.rank(method="first") if k > 3 else pd.Series(dtype=float)
                ctx = dict(gt=gt, one_day=one_day, progress=k / n_st, last=k == n_st, quality_z=q_z,
                           sprint_teams=sprint_teams, gc_rank=gc_rank, prev_break=prev_break)
                out = simulate_stage(rng, st, sl, race, ctx)
                row.update(won_how=out["won_how"], finished=True)
                tables["stages"].append(row)
                pts = np.where(out["rank"] <= 15, np.array(STAGE_POINTS + [0])[np.minimum(out["rank"], 16) - 1], 0)
                tables["results"] += [dict(stage_id=stage_id, rider_id=r, team=t, rank=int(rk) if s == "DF" else None,
                                           status=s, age=year - by, breakaway_kms=float(b), pcs_points=int(p) * (3 if one_day else 1))
                                      for r, t, rk, s, by, b, p in zip(sl["rider_id"], sl["team"], out["rank"], out["status"],
                                                                       sl["birth_year"], out["breakaway"], pts)]
                tables["climbs"] += [dict(stage_id=stage_id, **c) for c in out["climbs"]]
                if not one_day:
                    gc_time = gc_time + pd.Series(out["gc_s"], index=sl["rider_id"].values)
                    g = gc_time.sort_values()
                    tables["gc"] += [dict(stage_id=stage_id, rider_id=r, gc_rank=i + 1, gc_gap_s=int(v - g.iloc[0]))
                                     for i, (r, v) in enumerate(g.items())]
                prev_break = set(sl["rider_id"][out["in_break"]])
    out = {k: pd.DataFrame(v) for k, v in tables.items()}
    # rider pages: specialty points and points per season, as PCS computes them from results
    res = out["results"].merge(out["stages"][["stage_id", "year", "profile_icon", "stage_type", "slug"]], on="stage_id")
    res["spec"] = np.select([res["stage_type"] == "ITT", res["slug"].isin(cal.loc[cal.kind == "one_day", "pcs_slug"]),
                             res["profile_icon"].isin(["p4", "p5"]), res["profile_icon"].isin(["p2", "p3"])],
                            ["time_trial", "one_day_races", "climber", "hills"], "sprint")
    spec = res.pivot_table(index="rider_id", columns="spec", values="pcs_points", aggfunc="sum", fill_value=0)
    riders_out = riders[["rider_id", "nationality", "height", "weight"]].copy()
    riders_out["name"] = riders["rider_id"].str.split("-", n=1).str[1].str.replace("-", " ").str.title()
    riders_out["birthdate"] = riders["birth_year"].astype(str) + "-6-15"
    for s in ["one_day_races", "gc", "time_trial", "sprint", "climber", "hills"]:
        riders_out[f"spec_{s}"] = riders_out["rider_id"].map(spec[s] if s in spec else {}).fillna(0).astype(int)
    seasons = res.groupby(["rider_id", "year"])["pcs_points"].sum().reset_index().rename(
        columns={"year": "season", "pcs_points": "points"})
    seasons["rank"] = seasons.groupby("season")["points"].rank(ascending=False, method="min").astype(int)
    seasons["team_class"] = seasons["rider_id"].map(riders.set_index("rider_id")["team_class"])
    out["riders"] = riders_out
    out["rider_seasons"] = seasons
    return out


def write(out_dir=config.DEMO, seed=3):
    out = generate(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(out_dir / (f"{k}.csv.gz" if k in ("results", "gc", "startlists") else f"{k}.csv"), index=False)
    (out_dir / "DEMO_TODAY").write_text(DEMO_TODAY.date().isoformat())
    st = out["stages"]
    print(f"demo: {len(out['races'])} editions, {len(st)} stages ({st['finished'].sum()} raced), "
          f"{len(out['results'])} results, {len(out['riders'])} riders -> {out_dir}")
