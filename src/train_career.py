"""Train the career-trajectory models, one per cutoff age.

For each cutoff age:
  1. build features at the cutoff and the rider x future-age panel
  2. hold out 20% of riders as a test set (whole careers, never seen in training)
  3. compare OLS / Ridge / Lasso / Elastic Net with 5-fold *rider-grouped* CV
  4. refit the winner, score it on the test riders
  5. statsmodels OLS with cluster-robust (by rider) standard errors -> coefficients,
     95% CI, p-values; VIF for multicollinearity
  6. split-conformal intervals per horizon from out-of-fold residuals
  7. logistic regression for P(still racing)

    python -m src.train_career --data data/careers
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.career_model import (CUTOFFS, LABELS, MAX_AGE, available, build_pipeline,
                              cutoff_features, make_panel, model_specs, survival_pipeline)

rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))


def grouped_cv(spec, X, y, groups, folds):
    """Out-of-fold predictions for every hyper-parameter setting; keep the best."""
    design, est, grid = spec
    best = None
    for params in grid:
        oof = np.zeros(len(y))
        for tr, va in folds:
            pipe = build_pipeline(design, est, params).fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict(X.iloc[va])
        score = rmse(y, oof)
        if best is None or score < best[0]:
            best = (score, params, oof)
    return best


def conformal(resid, horizon, n_h, lo=0.10, hi=0.90):
    """Residual quantiles per horizon, pooling neighbours where data is thin."""
    q_lo, q_hi = [], []
    for h in range(1, n_h + 1):
        w = 0
        while True:
            m = np.abs(horizon - h) <= w
            if m.sum() >= 150 or w > n_h:
                break
            w += 1
        r = resid[m] if m.any() else resid
        q_lo.append(float(np.quantile(r, lo)))
        q_hi.append(float(np.quantile(r, hi)))
    return q_lo, q_hi


def inference_table(train, feats):
    """Interpretable OLS (main effects, standardised) with rider-clustered SEs."""
    num = vif_prune(feats, available([c for c in LABELS if c in feats], feats))
    cat = available(["nationality", "specialty", "background", "current_level"], feats)
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False,
                              min_frequency=15), cat),
        ("age", SplineTransformer(n_knots=5, degree=3, include_bias=False), ["target_age"]),
    ], verbose_feature_names_out=False)
    X = pd.DataFrame(ct.fit_transform(train), columns=ct.get_feature_names_out())
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(train["target_lp"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(train["rider_id"])[0]})
    ci = fit.conf_int()
    tab = pd.DataFrame({"coef": fit.params, "se": fit.bse, "ci_low": ci[0], "ci_high": ci[1],
                        "p_value": fit.pvalues}).drop(index="const")
    tab = tab[~tab.index.str.startswith("target_age")]
    tab["label"] = [LABELS.get(i, i.replace("_", " ")) for i in tab.index]
    return tab.reset_index(names="feature"), float(fit.rsquared), float(fit.rsquared_adj)


def _vif(feats, cols):
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(feats[cols]))
    X = sm.add_constant(X)
    with np.errstate(divide="ignore"):
        return np.array([variance_inflation_factor(X, i + 1) for i in range(len(cols))])


def vif_prune(feats, cols, threshold=10.0):
    """Drop the most collinear variable until every VIF is below the threshold."""
    cols = [c for c in cols if feats[c].nunique() > 1]
    while True:
        v = _vif(feats, cols)
        if v.max() < threshold:
            return cols
        cols.pop(int(np.argmax(v)))


def vif_table(feats):
    num = [c for c in available([c for c in LABELS if c in feats], feats) if feats[c].nunique() > 1]
    kept = set(vif_prune(feats, num))
    return pd.DataFrame({"feature": num, "label": [LABELS.get(c, c) for c in num],
                         "vif": np.minimum(_vif(feats, num), 1e6),
                         "in_inference_model": [c in kept for c in num]}).sort_values("vif", ascending=False)


def train_cutoff(riders, seasons, cutoff, test_riders, log):
    t0 = time.time()
    feats = cutoff_features(riders, seasons, cutoff)
    panel = make_panel(feats, seasons, cutoff)
    is_test = panel["rider_id"].isin(test_riders)
    act = panel[panel["active"]].reset_index(drop=True)
    tr, te = act[~act["rider_id"].isin(test_riders)], act[act["rider_id"].isin(test_riders)]
    y_tr, y_te = tr["target_lp"].to_numpy(), te["target_lp"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(tr, y_tr, tr["rider_id"]))
    log(f"cutoff {cutoff}: {len(feats)} riders, {len(act)} active rider-seasons "
        f"({len(tr)} train / {len(te)} test)")

    specs = model_specs(feats)
    comparison, best = [], None
    for name, spec in specs.items():
        cv_rmse, params, oof = grouped_cv(spec, tr, y_tr, tr["rider_id"], folds)
        pipe = build_pipeline(spec[0], spec[1], params).fit(tr, y_tr)
        pred = pipe.predict(te)
        coefs = pipe["model"].coef_
        row = dict(model=name, params=params, cv_rmse=cv_rmse, cv_r2=float(r2_score(y_tr, oof)),
                   test_rmse=rmse(y_te, pred), test_r2=float(r2_score(y_te, pred)),
                   n_columns=int(len(coefs)), n_nonzero=int((np.abs(coefs) > 1e-8).sum()))
        comparison.append(row)
        log(f"   {name:<34} CV RMSE {cv_rmse:.3f}  test R2 {row['test_r2']:.3f}  "
            f"({row['n_nonzero']}/{row['n_columns']} coefs) {params}")
        if best is None or cv_rmse < best[0]:
            best = (cv_rmse, name, params, oof, pipe, pred)
    _, best_name, best_params, oof, model, pred = best  # model: fit on train riders only

    horizon_tr = (tr["target_age"] - cutoff).to_numpy()
    q_lo, q_hi = conformal(y_tr - oof, horizon_tr, MAX_AGE - cutoff)
    h_te = (te["target_age"] - cutoff).clip(1, len(q_lo)).to_numpy().astype(int) - 1
    covered = (y_te >= pred + np.array(q_lo)[h_te]) & (y_te <= pred + np.array(q_hi)[h_te])

    surv = survival_pipeline(feats).fit(panel[~is_test], panel.loc[~is_test, "active"])
    p_act = surv.predict_proba(panel[is_test])[:, 1]

    # career-level check: predicted vs actual best future season
    te_c = te.assign(pred=pred)
    per = te_c.groupby("rider_id").agg(actual=("target_lp", "max"), pred=("pred", "max"),
                                       n=("target_lp", "size"))
    per = per[per["n"] >= 5]

    coef_tab, ols_r2, ols_r2_adj = inference_table(tr, feats)
    names = model["design"].get_feature_names_out()
    metrics = dict(
        cutoff=cutoff, best_model=best_name, best_params=best_params,
        n_riders=int(len(feats)), n_train_rows=int(len(tr)), n_test_rows=int(len(te)),
        test_rmse=rmse(y_te, pred), test_r2=float(r2_score(y_te, pred)),
        interval_coverage_80=float(covered.mean()),
        survival_auc=float(roc_auc_score(panel.loc[is_test, "active"], p_act)),
        survival_brier=float(brier_score_loss(panel.loc[is_test, "active"], p_act)),
        peak_spearman=float(spearmanr(per["actual"], per["pred"]).statistic),
        peak_r2=float(r2_score(per["actual"], per["pred"])), peak_n=int(len(per)),
        inference_ols_r2=ols_r2, inference_ols_r2_adj=ols_r2_adj,
        n_features_raw=int(len(available(list(LABELS), feats)) + 4), n_design_columns=int(len(names)),
        seconds=round(time.time() - t0, 1),
    )
    log(f"   -> {best_name}: test R2 {metrics['test_r2']:.3f}, 80% interval coverage "
        f"{metrics['interval_coverage_80']:.1%}, survival AUC {metrics['survival_auc']:.3f}, "
        f"peak Spearman {metrics['peak_spearman']:.3f}  [{metrics['seconds']}s]")

    diag = te_c.assign(horizon=te_c["target_age"] - cutoff, resid=te_c["target_lp"] - te_c["pred"])
    diag = diag[["rider_id", "target_age", "horizon", "target_lp", "pred", "resid"]]
    # final models for the app: refit on every rider (train + test)
    final = build_pipeline(*specs[best_name][:2], best_params).fit(act, act["target_lp"])
    final_surv = survival_pipeline(feats).fit(panel, panel["active"])
    resid_var = float(np.var(y_tr - oof))
    return dict(
        cutoff=cutoff, model=final, survival=final_surv, q_lo=q_lo, q_hi=q_hi,
        holdout_model=model, holdout_survival=surv,
        test_riders=sorted(set(feats["rider_id"]) & test_riders),
        resid_var=resid_var, metrics=metrics, comparison=pd.DataFrame(comparison),
        coef_table=coef_tab, vif=vif_table(feats),
        penalised_coefs=pd.DataFrame({"term": final["design"].get_feature_names_out(),
                                      "coef": final["model"].coef_}),
        diagnostics=diag.sample(min(len(diag), 4000), random_state=0),
        per_rider=per.reset_index(),
    )


def main(data_dir, out_dir, cutoffs):
    d = Path(data_dir)
    riders = pd.read_csv(d / "riders.csv")
    seasons = pd.read_csv(d / "seasons.csv")
    rng = np.random.default_rng(0)
    ids = riders["rider_id"].to_numpy()
    test_riders = set(rng.choice(ids, size=len(ids) // 5, replace=False))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c in cutoffs:
        art = train_cutoff(riders, seasons, c, test_riders, print)
        art["data_dir"] = Path(data_dir).as_posix()
        joblib.dump(art, out / f"cutoff_{c}.joblib", compress=3)
        summary[c] = art["metrics"]
    (out / "summary.json").write_text(json.dumps({"data_dir": Path(data_dir).as_posix(),
                                                  "cutoffs": summary}, indent=2, default=str))
    print(f"Saved {len(cutoffs)} models to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/careers")
    ap.add_argument("--models", default="models/career")
    ap.add_argument("--cutoffs", default=",".join(map(str, CUTOFFS)))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")  # convergence / VIF division notices
    main(args.data, args.models, [int(c) for c in args.cutoffs.split(",")])
