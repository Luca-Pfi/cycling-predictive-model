"""Career-trajectory model: predict a rider's PCS points at every future age.

The idea: freeze a rider's file at a *cutoff age* (21-25) and use everything
known by then (lab tests, junior and U23 results, pro results so far, team,
injuries, nationality, ...) to predict log PCS points at each later age.

The model is a penalised linear regression on a long "rider x future age" panel:
  * main effects of ~45 variables (standardised, imputed, one-hot encoded)
  * a cubic spline of the target age (the average career curve)
  * spline x feature interactions, so each variable can bend the curve
    (e.g. strong U23 results raise the early career, late starters peak later)
That gives ~300 columns. OLS, Ridge, Lasso and Elastic Net are compared with
rider-grouped cross-validation. Two more pieces go with it:
  * a logistic regression for P(still racing at that age), which handles
    survivorship (weak riders retire, their missing seasons are not random)
  * split-conformal prediction intervals per horizon (years ahead)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

CUTOFFS = [21, 22, 23, 24, 25]
MAX_AGE = 38

STATIC_NUMERIC = ["height_cm", "weight_kg", "bmi", "vo2max", "ftp_wkg_u23", "sprint_peak_w",
                  "sprint_wkg", "started_cycling_age", "junior_worlds_top10",
                  "junior_nc_points_log", "devo_team"]
DYNAMIC_NUMERIC = ["lp_now", "lp_prev", "lp_prev2", "lp_mean", "lp_max", "lp_slope",
                   "career_points_log", "wins_total", "top10s_total", "top10_rate",
                   "race_days_per_season", "gt_starts_total", "wt_seasons", "pro_seasons",
                   "team_changes", "injuries", "injured_now", "u23_wins_total",
                   "u23_podiums_total", "avenir_score", "u23_worlds_score",
                   "seasons_observed"]
CATEGORICAL = ["nationality", "specialty", "background", "current_level"]
# variables allowed to change the *shape* of the career curve
INTERACT = ["lp_now", "lp_slope", "lp_max", "career_points_log", "ftp_wkg_u23", "vo2max",
            "started_cycling_age", "wt_seasons", "u23_wins_total", "avenir_score",
            "specialty_gc", "specialty_sprinter", "specialty_climber", "specialty_rouleur",
            "current_level_wt"]
LABELS = {
    "lp_now": "Log points, cutoff season", "lp_prev": "Log points, season before",
    "lp_prev2": "Log points, 2 seasons before", "lp_mean": "Mean log points so far",
    "lp_max": "Best season so far (log)", "lp_slope": "Trend of log points (last 3 seasons)",
    "career_points_log": "Career points so far (log)", "wins_total": "Wins so far",
    "top10s_total": "Top-10s so far", "top10_rate": "Top-10s per race day",
    "race_days_per_season": "Race days per season", "gt_starts_total": "Grand Tour starts",
    "wt_seasons": "Seasons in a WorldTeam", "pro_seasons": "Seasons in a Pro/WorldTeam",
    "team_changes": "Team-level changes", "injuries": "Injury seasons",
    "injured_now": "Injured in cutoff season", "u23_wins_total": "U23 wins",
    "u23_podiums_total": "U23 podiums", "avenir_score": "Tour de l'Avenir (31 - best rank)",
    "u23_worlds_score": "U23 Worlds (41 - best rank)", "seasons_observed": "Seasons on record",
    "height_cm": "Height (cm)", "weight_kg": "Weight (kg)", "bmi": "BMI", "vo2max": "VO2max",
    "ftp_wkg_u23": "FTP at 19 (W/kg)", "sprint_peak_w": "5-s sprint power (W)",
    "sprint_wkg": "5-s sprint power (W/kg)", "started_cycling_age": "Age started cycling",
    "junior_worlds_top10": "Junior Worlds top 10", "junior_nc_points_log": "Junior Nations Cup points (log)",
    "devo_team": "Rode for a development team",
}


# --------------------------------------------------------------------- features
def cutoff_features(riders, seasons, cutoff, last_season=None):
    """One row per rider who is still racing at `cutoff`, using seasons up to it."""
    last_season = last_season or int(seasons["season"].max())
    s = seasons[seasons["age"] <= cutoff].copy()
    s["lp"] = np.log1p(s["points"])
    g = s.sort_values("age").groupby("rider_id")
    at = s[s["age"] == cutoff].set_index("rider_id")
    eligible = riders.set_index("rider_id")
    eligible = eligible[eligible.index.isin(at.index) & (eligible["birth_year"] + cutoff <= last_season)]

    f = pd.DataFrame(index=eligible.index)
    lp_by_age = s.pivot_table(index="rider_id", columns="age", values="lp")
    f["lp_now"] = lp_by_age.get(cutoff)
    f["lp_prev"] = lp_by_age.get(cutoff - 1)
    f["lp_prev2"] = lp_by_age.get(cutoff - 2)
    f["lp_mean"] = g["lp"].mean()
    f["lp_max"] = g["lp"].max()
    last3 = s[s["age"] > cutoff - 3]
    a = last3["age"] - last3.groupby("rider_id")["age"].transform("mean")
    f["lp_slope"] = (a * last3["lp"]).groupby(last3["rider_id"]).sum() / (a**2).groupby(last3["rider_id"]).sum().replace(0, np.nan)
    f["career_points_log"] = np.log1p(g["points"].sum())
    for col, src in [("wins_total", "wins"), ("top10s_total", "top10s"), ("gt_starts_total", "gt_starts"),
                     ("injuries", "injured"), ("u23_wins_total", "u23_wins"), ("u23_podiums_total", "u23_podiums")]:
        if src in s:
            f[col] = g[src].sum()
    if "race_days" in s:
        f["race_days_per_season"] = g["race_days"].mean()
        if "top10s" in s:
            f["top10_rate"] = g["top10s"].sum() / g["race_days"].sum().clip(lower=1)
    if "team_level" in s:
        f["wt_seasons"] = g["team_level"].agg(lambda x: (x == "wt").sum())
        f["pro_seasons"] = g["team_level"].agg(lambda x: x.isin(["pro", "wt"]).sum())
        f["team_changes"] = g["team_level"].agg(lambda x: (x != x.shift()).sum() - 1)
        f["current_level"] = at["team_level"]
    if "injured" in s:
        f["injured_now"] = at["injured"]
    if "avenir_rank" in s:
        f["avenir_score"] = (31 - g["avenir_rank"].min()).fillna(0)
    if "u23_worlds_rank" in s:
        f["u23_worlds_score"] = (41 - g["u23_worlds_rank"].min()).fillna(0)
    f["seasons_observed"] = g.size()

    r = eligible
    for col in ["height_cm", "weight_kg", "vo2max", "ftp_wkg_u23", "sprint_peak_w",
                "started_cycling_age", "junior_worlds_top10", "devo_team",
                "nationality", "specialty", "background"]:
        if col in r:
            f[col] = r[col]
    if {"height_cm", "weight_kg"} <= set(r):
        f["bmi"] = r["weight_kg"] / (r["height_cm"] / 100) ** 2
    if {"sprint_peak_w", "weight_kg"} <= set(r):
        f["sprint_wkg"] = r["sprint_peak_w"] / r["weight_kg"]
    if "junior_nc_points" in r:
        f["junior_nc_points_log"] = np.log1p(r["junior_nc_points"])
    f["birth_year"] = r["birth_year"]
    f["cutoff"] = cutoff
    return f.reset_index()


def make_panel(feats, seasons, cutoff, last_season=None):
    """Long table: one row per rider and future age that has already happened."""
    last_season = last_season or int(seasons["season"].max())
    s = seasons.set_index(["rider_id", "age"])["points"]
    rows = []
    for r in feats[["rider_id", "birth_year"]].itertuples(index=False):
        for age in range(cutoff + 1, min(MAX_AGE, last_season - r.birth_year) + 1):
            pts = s.get((r.rider_id, age))
            rows.append((r.rider_id, age, pts is not None, np.log1p(pts) if pts is not None else np.nan))
    panel = pd.DataFrame(rows, columns=["rider_id", "target_age", "active", "target_lp"])
    return panel.merge(feats, on="rider_id")


# ----------------------------------------------------------------------- design
class CareerDesign(BaseEstimator, TransformerMixin):
    """Main effects + spline(target age) + spline x feature interactions."""

    def __init__(self, numeric, categorical, interact=(), n_knots=5):
        self.numeric = numeric
        self.categorical = categorical
        self.interact = interact
        self.n_knots = n_knots

    def fit(self, X, y=None):
        parts = []
        if self.numeric:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                               StandardScaler()), list(self.numeric)))
        if self.categorical:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=15,
                                               sparse_output=False), list(self.categorical)))
        self.ct_ = ColumnTransformer(parts, verbose_feature_names_out=False).fit(X)
        self.base_names_ = [n.replace("missingindicator_", "missing_") for n in self.ct_.get_feature_names_out()]
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False,
                                         extrapolation="linear").fit(X[["target_age"]])
        self.n_spline_ = self.spline_.transform(X[["target_age"]].iloc[:1]).shape[1]
        self.inter_idx_ = [self.base_names_.index(n) for n in self.interact if n in self.base_names_]
        return self

    def transform(self, X):
        base = self.ct_.transform(X)
        S = self.spline_.transform(X[["target_age"]])
        inter = [base[:, [i]] * S for i in self.inter_idx_]
        return np.hstack([base, S, *inter])

    def get_feature_names_out(self, input_features=None):
        spl = [f"age_spline_{k}" for k in range(self.n_spline_)]
        inter = [f"{self.base_names_[i]} x {s}" for i in self.inter_idx_ for s in spl]
        return np.array(self.base_names_ + spl + inter)


def available(cols, feats):
    return [c for c in cols if c in feats and feats[c].notna().any()]


def model_specs(feats):
    num = available(STATIC_NUMERIC + DYNAMIC_NUMERIC, feats)
    cat = available(CATEGORICAL, feats)
    full = dict(numeric=num, categorical=cat, interact=INTERACT)
    main = dict(numeric=num, categorical=cat, interact=())
    base = dict(numeric=available(["lp_now"], feats), categorical=(), interact=())
    return {  # name: (design kwargs, estimator, hyper-parameter grid)
        "Baseline OLS (age + current points)": (base, LinearRegression(), [{}]),
        "OLS, main effects": (main, LinearRegression(), [{}]),
        "OLS + age interactions": (full, LinearRegression(), [{}]),
        "Ridge + age interactions": (full, Ridge(), [{"alpha": a} for a in [0.1, 1, 10, 30, 100, 300]]),
        "Lasso + age interactions": (full, Lasso(max_iter=5000, tol=1e-3),
                                     [{"alpha": a} for a in [0.0001, 0.0003, 0.001, 0.003, 0.01]]),
        "Elastic Net + age interactions": (full, ElasticNet(max_iter=5000, tol=1e-3),
                                           [{"alpha": a, "l1_ratio": r} for a in [0.0003, 0.001, 0.003]
                                            for r in [0.2, 0.5, 0.8]]),
    }


def build_pipeline(design_kwargs, estimator, params):
    return Pipeline([("design", CareerDesign(**design_kwargs)),
                     ("model", clone(estimator).set_params(**params))])


def survival_pipeline(feats):
    spec = model_specs(feats)["OLS + age interactions"][0]
    return Pipeline([("design", CareerDesign(**spec)),
                     ("model", LogisticRegression(C=0.3, max_iter=3000))])


# -------------------------------------------------------------------- inference
def project(art, feats, ages=None, holdout=False):
    """Projected career curve(s) with conformal intervals and P(still racing).

    holdout=True uses the models fitted without the test riders (for honest backtests).
    """
    c = art["cutoff"]
    ages = ages if ages is not None else range(c + 1, MAX_AGE + 1)
    grid = feats.merge(pd.DataFrame({"target_age": list(ages)}), how="cross")
    prefix = "holdout_" if holdout else ""
    grid["mu"] = art[prefix + "model"].predict(grid)
    grid["p_active"] = art[prefix + "survival"].predict_proba(grid)[:, 1]
    h = (grid["target_age"] - c).clip(1, len(art["q_lo"])).astype(int) - 1
    grid["lo"] = grid["mu"] + np.asarray(art["q_lo"])[h]
    grid["hi"] = grid["mu"] + np.asarray(art["q_hi"])[h]
    for k in ["mu", "lo", "hi"]:
        grid[f"{k}_pts"] = np.expm1(grid[k].clip(lower=0))
    grid["expected_pts"] = grid["p_active"] * np.expm1(grid["mu"].clip(lower=0) + art["resid_var"] / 2)
    return grid


def summarise(curves):
    """Peak points, peak age and expected remaining points per rider."""
    idx = curves.groupby("rider_id")["mu"].idxmax()
    peak = curves.loc[idx, ["rider_id", "target_age", "mu_pts", "lo_pts", "hi_pts"]]
    peak.columns = ["rider_id", "peak_age", "peak_pts", "peak_lo", "peak_hi"]
    total = curves.groupby("rider_id")["expected_pts"].sum().rename("expected_career_pts")
    return peak.merge(total, on="rider_id")
