"""Pre-race models: does the break win, who is in it, who wins; walk-forward tests.

    P(win_i) = P(break wins) * P(i in break) * P(i wins the break | it survives)
             + P(break caught) * P(i wins otherwise)

Fair decimal odds are 1 / P. Every model is evaluated walk-forward: trained on all
races before a season, tested on that season, against a simple baseline.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from cyclepredict import config, data, features
from cyclepredict.features import RIDER_FEATURES, STAGE_FEATURES

MEMBER_FEATURES = RIDER_FEATURES + STAGE_FEATURES
WIN_FEATURES = RIDER_FEATURES + STAGE_FEATURES


def _clf(**kw):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=40, random_state=0, **kw)


def prepare(t=None):
    """Load, label and build features for every stage (raced and upcoming)."""
    if t is None:
        t, _ = data.load()
    t = data.label_breaks(t)
    st = t["stages"]
    hist, shist = features.rider_history(t), features.stage_history(st)
    rows, stages = features.build(t, st["stage_id"].tolist(), hist, shist)
    lab = st.set_index("stage_id")
    stages = stages.merge(lab[["date", "year", "slug", "race_id", "labelled", "break_won", "break_size",
                               "finished", "stage_type", "winner"]], left_on="stage_id", right_index=True)
    res = t["results"].set_index(["stage_id", "rider_id"])
    key = list(zip(rows["stage_id"], rows["rider_id"]))
    rows["in_break"] = res["in_break"].reindex(key).fillna(False).astype(bool).to_numpy()
    rows["won"] = (res["rank"].reindex(key) == 1).to_numpy()
    rows = rows.merge(stages[["stage_id", "labelled", "break_won", "break_size", "finished", "stage_type", "year"]],
                      on="stage_id", how="left")
    return t, rows, stages


class Predictor:
    """The five models of one training run."""

    def fit(self, rows, stages):
        s = stages[stages["labelled"].fillna(False).astype(bool)]
        self.break_win = CalibratedClassifierCV(_clf(), method="isotonic", cv=5).fit(
            s[STAGE_FEATURES], s["break_won"].astype(int))
        self.break_size = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0,
                                                        loss="poisson").fit(s[STAGE_FEATURES], s["break_size"])
        lab = rows[rows["labelled"].fillna(False).astype(bool)]
        self.member = _clf().fit(lab[MEMBER_FEATURES], lab["in_break"].astype(int))
        fin = lab[lab["break_won"].fillna(False).astype(bool) & lab["in_break"]]
        self.finale = _clf().fit(fin[WIN_FEATURES], fin["won"].astype(int))
        fav = rows[rows["finished"].fillna(False).astype(bool) & ~rows["break_won"].fillna(False).astype(bool)
                   & ~(rows["labelled"].fillna(False).astype(bool) & rows["in_break"])]
        self.favourite = _clf().fit(fav[WIN_FEATURES], fav["won"].astype(int))
        return self

    def predict(self, rows, stages):
        """Per stage: P(break wins), break size; per rider: P(in break), P(win), split by route."""
        st = stages.set_index("stage_id")
        is_rr = st["stage_type"].reindex(rows["stage_id"]).ne("ITT").to_numpy()
        out = rows[["stage_id", "rider_id", "team"]].copy()
        p_bw = pd.Series(self.break_win.predict_proba(st[STAGE_FEATURES])[:, 1], index=st.index)
        p_bw[st["stage_type"] == "ITT"] = 0.0
        size = pd.Series(self.break_size.predict(st[STAGE_FEATURES]), index=st.index)
        m = self.member.predict_proba(rows[MEMBER_FEATURES])[:, 1] * is_rr
        # rescale membership so it adds up to the expected break size of the stage
        tot = pd.Series(m).groupby(rows["stage_id"].to_numpy()).transform("sum").to_numpy()
        m = np.clip(m * size.reindex(rows["stage_id"]).to_numpy() / np.maximum(tot, 1e-9), 0, 0.97)
        fo = self.finale.predict_proba(rows[WIN_FEATURES])[:, 1]
        fo = fo / (1 - fo + 1e-9)
        mo = pd.Series(m * fo).groupby(rows["stage_id"].to_numpy()).transform("sum").to_numpy()
        q = m * fo / (fo + (mo - m * fo) + 1e-12)  # P(i wins | break wins)
        q = q / np.maximum(pd.Series(q).groupby(rows["stage_id"].to_numpy()).transform("sum").to_numpy(), 1e-12)
        r = self.favourite.predict_proba(rows[WIN_FEATURES])[:, 1]
        r = r / pd.Series(r).groupby(rows["stage_id"].to_numpy()).transform("sum").to_numpy()
        pb = p_bw.reindex(rows["stage_id"]).to_numpy()
        out["p_break"] = m
        out["p_win_if_break"] = q
        out["p_win_if_caught"] = r
        out["p_win"] = pb * q + (1 - pb) * r
        stage_out = pd.DataFrame({"stage_id": st.index, "p_break_wins": p_bw.to_numpy(),
                                  "expected_break_size": size.clip(lower=1).to_numpy()})
        return out, stage_out


def _stage_metrics(pred_r, pred_s, rows, stages):
    """Accuracy of one batch of predictions against what happened."""
    s = stages.merge(pred_s, on="stage_id")
    L = s[s["labelled"].fillna(False).astype(bool)]
    r = rows[["stage_id", "rider_id", "in_break", "won", "labelled", "break_size"]].merge(pred_r, on=["stage_id", "rider_id"])
    m = {}
    if len(L) > 5 and L["break_won"].nunique() > 1:
        y, p = L["break_won"].astype(int), L["p_break_wins"].clip(1e-4, 1 - 1e-4)
        base = L["race_bw_rate"].clip(1e-4, 1 - 1e-4)
        m.update(bw_brier=brier_score_loss(y, p), bw_brier_baseline=brier_score_loss(y, base),
                 bw_logloss=log_loss(y, p), bw_auc=roc_auc_score(y, p), bw_auc_baseline=roc_auc_score(y, base),
                 n_break_stages=int(len(L)))
    rl = r[r["labelled"].fillna(False).astype(bool)]
    if len(rl) and rl["in_break"].any():
        m["member_auc"] = roc_auc_score(rl["in_break"], rl["p_break"])
        base = rows.loc[rows["labelled"].fillna(False).astype(bool)].set_index(["stage_id", "rider_id"])["break_rate"]
        m["member_auc_baseline"] = roc_auc_score(rl["in_break"], base.reindex(list(zip(rl["stage_id"], rl["rider_id"]))).fillna(0))
        hits, base_hits = [], []
        for _, g in rl.groupby("stage_id"):
            k = int(g["in_break"].sum())
            if k:
                hits.append(g.nlargest(k, "p_break")["in_break"].mean())
        m["member_precision_at_k"] = float(np.mean(hits))
    fin = r.merge(stages[["stage_id", "finished"]], on="stage_id")
    fin = fin[fin["finished"].fillna(False).astype(bool)]
    won = fin[fin["won"].fillna(False).astype(bool)]
    if len(won):
        rank = fin.groupby("stage_id")["p_win"].rank(ascending=False, method="first")
        wr = rank[fin["won"].fillna(False).astype(bool)]
        m.update(win_logloss=float(-np.log(won["p_win"].clip(1e-4)).mean()), winner_top1=float((wr <= 1).mean()),
                 winner_top3=float((wr <= 3).mean()), winner_top10=float((wr <= 10).mean()), n_stages=int(len(won)))
        form = rows.set_index(["stage_id", "rider_id"])["pct_grp"].reindex(list(zip(fin["stage_id"], fin["rider_id"]))).to_numpy()
        brank = pd.Series(form).groupby(fin["stage_id"].to_numpy()).rank(ascending=False, method="first").to_numpy()
        m["winner_top3_baseline"] = float((brank[fin["won"].fillna(False).astype(bool).to_numpy()] <= 3).mean())
    return m


def walk_forward(rows, stages, test_years, log=print):
    """Train on everything before each season, test on the season."""
    preds_r, preds_s, per_year = [], [], {}
    for y in test_years:
        tr = stages["date"] < pd.Timestamp(f"{y}-01-01")
        te = (stages["year"] == y) & stages["finished"].fillna(False).astype(bool)
        tr_ids, te_ids = set(stages.loc[tr, "stage_id"]), set(stages.loc[te, "stage_id"])
        if not te_ids or len(tr_ids) < 100:
            continue
        p = Predictor().fit(rows[rows["stage_id"].isin(tr_ids)], stages[stages["stage_id"].isin(tr_ids)])
        r, s = p.predict(rows[rows["stage_id"].isin(te_ids)], stages[stages["stage_id"].isin(te_ids)])
        per_year[y] = _stage_metrics(r, s, rows, stages)
        log(f"  {y}: " + ", ".join(f"{k} {v:.3f}" for k, v in per_year[y].items() if not k.startswith("n_")))
        preds_r.append(r)
        preds_s.append(s)
    return pd.concat(preds_r), pd.concat(preds_s), per_year


def per_race(pred_r, pred_s, rows, stages):
    """Accuracy per race, so every race (not only Grand Tours) gets checked."""
    out = []
    for slug, s in stages[stages["stage_id"].isin(pred_s["stage_id"])].groupby("slug"):
        ids = set(s["stage_id"])
        m = _stage_metrics(pred_r[pred_r["stage_id"].isin(ids)], pred_s[pred_s["stage_id"].isin(ids)],
                           rows[rows["stage_id"].isin(ids)], s)
        out.append(dict(slug=slug, **m))
    return pd.DataFrame(out)


def train(log=print, test_years=None):
    t0 = time.time()
    t, rows, stages = prepare()
    log(f"features: {len(rows):,} rider-stage rows, {len(stages):,} stages [{time.time() - t0:.0f}s]")
    years = sorted(stages.loc[stages["finished"].fillna(False).astype(bool), "year"].unique())
    test_years = test_years or years[-3:]
    log("walk-forward evaluation:")
    pr, ps, per_year = walk_forward(rows, stages, test_years, log)
    races = per_race(pr, ps, rows, stages)
    done = stages["finished"].fillna(False).astype(bool)
    final = Predictor().fit(rows[rows["stage_id"].isin(stages.loc[done, "stage_id"])], stages[done])
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODELS / "predictor.joblib", compress=3)
    overall = _stage_metrics(pr, ps, rows, stages)
    meta = dict(trained_at=pd.Timestamp.now().isoformat(timespec="seconds"), data=str(config.data_dir().name),
                demo=config.is_demo(), test_years=[int(y) for y in test_years], overall=overall,
                per_year={int(k): v for k, v in per_year.items()},
                n_stages=int(done.sum()), n_labelled=int(stages["labelled"].fillna(False).sum()))
    (config.MODELS / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    races.to_csv(config.MODELS / "per_race.csv", index=False)
    ps.merge(stages[["stage_id", "slug", "date", "labelled", "break_won", "break_size"]], on="stage_id").to_csv(
        config.MODELS / "backtest_stages.csv", index=False)
    from cyclepredict import predict
    n = predict.backfill(pr, ps, stages)
    log(f"backtest previews stored for the track record: {n}")
    log(f"overall: " + ", ".join(f"{k} {v:.3f}" for k, v in overall.items() if not k.startswith("n_")))
    log(f"saved models -> {config.MODELS} [{time.time() - t0:.0f}s]")
    return meta
