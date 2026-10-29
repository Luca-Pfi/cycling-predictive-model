"""Train both models and evaluate them on the most recent season.

Model A: stage profile -> stage type (flat / hilly / mountain / itt)
Model B: stage profile + rider history -> probability of a top-10 finish

    python -m src.train --data data/sample
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from src.features import build_training_table, load_data, model_features, stage_features


def ranking_metrics(df, score_col):
    """Per-stage: share of the predicted top 10 that really finished top 10,
    and how often the real winner was among the 5 highest-scored riders."""
    p10, win5 = [], []
    for _, g in df.groupby("stage_id"):
        top = g.nlargest(10, score_col)
        p10.append(top["top10"].mean())
        win5.append((g.nlargest(5, score_col)["rank"] == 1).any())
    return float(np.mean(p10)), float(np.mean(win5))


def new_ranker():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=0)


def main(data_dir, model_dir):
    stages, riders, results = load_data(data_dir)
    STAGE_FEATURES = stage_features(stages)
    MODEL_FEATURES = model_features(STAGE_FEATURES)
    print(f"Stage features: {STAGE_FEATURES}\nBuilding leak-free features ...")
    table = build_training_table(stages, riders, results)
    table = table[table["n_starts"] >= 5]  # need some history to say anything

    test_year = stages["year"].max()
    test_ids = set(stages.loc[stages["year"] == test_year, "stage_id"])
    is_test = table["stage_id"].isin(test_ids)
    train, test = table[~is_test], table[is_test].copy()

    # Model A: stage type classifier
    st_train = stages[~stages["stage_id"].isin(test_ids)]
    st_test = stages[stages["stage_id"].isin(test_ids)]
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(st_train[STAGE_FEATURES], st_train["stage_type"])
    type_acc = accuracy_score(st_test["stage_type"], clf.predict(st_test[STAGE_FEATURES]))

    # Model B: top-10 probability
    model = new_ranker()
    model.fit(train[MODEL_FEATURES], train["top10"])
    test["p_top10"] = model.predict_proba(test[MODEL_FEATURES])[:, 1]

    p10, win5 = ranking_metrics(test, "p_top10")
    b_p10, b_win5 = ranking_metrics(test, "recent_pct")  # baseline: current form only
    sample = test.sample(min(len(test), 20000), random_state=0)
    imp = permutation_importance(model, sample[MODEL_FEATURES], sample["top10"],
                                 scoring="roc_auc", n_repeats=5, random_state=0)
    metrics = {
        "test_season": int(test_year),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "stage_type_accuracy": type_acc,
        "roc_auc": roc_auc_score(test["top10"], test["p_top10"]),
        "brier": brier_score_loss(test["top10"], test["p_top10"]),
        "precision_at_10": p10,
        "winner_in_top5": win5,
        "baseline_precision_at_10": b_p10,
        "baseline_winner_in_top5": b_win5,
        "feature_importance": dict(sorted(zip(MODEL_FEATURES, imp.importances_mean.tolist()),
                                          key=lambda kv: -kv[1])),
    }
    for k, v in metrics.items():
        if k != "feature_importance":
            print(f"{k:>26}: {v:.3f}" if isinstance(v, float) else f"{k:>26}: {v}")

    # refit on everything for the app
    clf.fit(stages[STAGE_FEATURES], stages["stage_type"])
    final = new_ranker().fit(table[MODEL_FEATURES], table["top10"])
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "stage_type_model.joblib")
    joblib.dump(final, out / "top10_model.joblib")
    test[["stage_id", "rider_id", "rank", "top10", "p_top10"]].to_csv(out / "test_predictions.csv", index=False)
    metrics["data_dir"] = Path(data_dir).as_posix()  # relative to the repo root
    metrics["stage_features"] = STAGE_FEATURES
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved models to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    main(args.data, args.models)
