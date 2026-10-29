"""Page 1: design a stage, see who suits it."""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import (MODEL_FEATURES, STAGE_FEATURES, STAGE_TYPES,
                          add_percentiles, features_for_new_stage, load_data)

MODELS = Path("models")
TYPE_COLORS = {"flat": "#2a9d8f", "hilly": "#e9c46a", "mountain": "#e76f51", "itt": "#577590"}
PRESETS = {  # distance, elevation, profile score, uphill finish, itt
    "Bunch sprint": (195, 900, 12, False, False),
    "Ardennes classic finale": (205, 3100, 120, True, False),
    "Rolling stage, breakaway day": (180, 2400, 75, False, False),
    "Alpine summit finish": (165, 4600, 330, True, False),
    "Mountain stage, descent finish": (175, 3900, 230, False, False),
    "Flat time trial": (32, 250, 10, False, True),
    "Uphill time trial": (18, 900, 90, True, True),
}

@st.cache_resource
def load_models():
    meta = json.loads((MODELS / "metrics.json").read_text())
    return (joblib.load(MODELS / "stage_type_model.joblib"),
            joblib.load(MODELS / "top10_model.joblib"), meta)


@st.cache_data
def load_all(data_dir):
    return load_data(data_dir)


if not (MODELS / "metrics.json").exists():
    st.error("No trained models found. Run `python -m src.make_sample_data` "
             "and `python -m src.train` first.")
    st.stop()

type_model, top10_model, meta = load_models()
stages, riders, results = load_all(meta["data_dir"])
season = int(stages["year"].max()) + 1
names = riders.set_index("rider_id")["name"]
teams = riders.set_index("rider_id")["team"]

st.title("🚴 Who suits this stage?")
st.caption(f"Data: `{meta['data_dir']}` · {len(stages)} stages · {riders.shape[0]} riders · "
           f"predictions for the {season} season")

tab_pred, tab_model, tab_rider = st.tabs(["Stage designer", "Model performance", "Rider explorer"])

# ---------------------------------------------------------------- designer
with st.sidebar:
    st.header("Design a stage")
    preset = st.selectbox("Start from a preset", list(PRESETS))
    d, e, ps, up, itt = PRESETS[preset]
    distance = st.slider("Distance (km)", 10, 260, d, key=f"d{preset}")
    elevation = st.slider("Elevation gain (m)", 0, 6000, e, step=50, key=f"e{preset}")
    profile = st.slider("Profile score (hardness)", 0, 450, ps, key=f"p{preset}",
                        help="PCS-style score: roughly the sum of (climb steepness² × length), "
                             "weighted towards climbs near the finish.")
    uphill = st.checkbox("Uphill finish", up, key=f"u{preset}")
    is_itt = st.checkbox("Individual time trial", itt, key=f"t{preset}")
    top_n = st.slider("Riders to show", 5, 30, 15)
    min_starts = st.slider("Min. starts in the last season", 1, 40, 10)

stage = dict(distance_km=distance, elevation_gain_m=elevation, profile_score=profile,
             uphill_finish=int(uphill), is_itt=int(is_itt))

last = stages[stages["year"] == season - 1]["stage_id"]
active = results[results["stage_id"].isin(last)]["rider_id"].value_counts()
active = active[active >= min_starts].index

feats, sim_row = features_for_new_stage(stage, stages, riders, results, active, season)
feats["p_top10"] = top10_model.predict_proba(feats[MODEL_FEATURES])[:, 1]
feats["name"] = feats["rider_id"].map(names)
feats["team"] = feats["rider_id"].map(teams)
board = feats.nlargest(top_n, "p_top10").iloc[::-1]

with tab_pred:
    type_probs = pd.Series(type_model.predict_proba(pd.DataFrame([stage])[STAGE_FEATURES])[0],
                           index=type_model.classes_).reindex(STAGE_TYPES).fillna(0)
    kind = type_probs.idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted stage type", kind.upper(), f"{type_probs.max():.0%} confidence",
              delta_color="off")
    c2.metric("Favourite", board.iloc[-1]["name"], f"{board.iloc[-1]['p_top10']:.0%} top-10",
              delta_color="off")
    c3.metric("Climbing per km", f"{elevation / max(distance, 1):.0f} m")
    c4.metric("Riders evaluated", len(feats))

    left, right = st.columns([2, 1])
    with left:
        fig = px.bar(board, x="p_top10", y="name", orientation="h", text="team",
                     hover_data={"sim_pct": ":.2f", "sim_top10": ":.2f", "recent_pct": ":.2f",
                                 "age": True, "p_top10": ":.1%", "name": False},
                     labels={"p_top10": "Probability of a top-10 finish", "name": ""},
                     title="Top candidates")
        fig.update_traces(marker_color="#e76f51", textposition="inside", insidetextanchor="start")
        fig.update_layout(height=32 * top_n + 120, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(type_probs.rename_axis("type").reset_index(name="prob"), x="type", y="prob",
                     color="type", color_discrete_map=TYPE_COLORS,
                     labels={"type": "", "prob": "Probability"}, title="Stage type")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=300)
        st.plotly_chart(fig, width="stretch")
        st.markdown("**How to read it**: features starting with `sim_` describe how the rider did "
                    "on past stages *similar to this one*; `recent_pct` is current form "
                    "(1.0 = always wins, 0.5 = mid-pack).")

    st.subheader("Most similar past stages")
    sim = stages.assign(similarity=sim_row).nlargest(8, "similarity")
    winners = results[results["rank"] == 1].set_index("stage_id")["rider_id"].map(names)
    sim["winner"] = sim["stage_id"].map(winners)
    st.dataframe(sim[["race", "date", "stage_type", *STAGE_FEATURES, "winner", "similarity"]]
                 .style.format({"similarity": "{:.2f}", "distance_km": "{:.0f}", "date": lambda x: f"{x:%Y-%m-%d}"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader(f"Evaluation on the held-out {meta['test_season']} season")
    st.caption("The model is trained on earlier seasons only and every feature uses results "
               "from *before* each stage, so these numbers are honest out-of-sample scores.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC (top-10)", f"{meta['roc_auc']:.3f}")
    c2.metric("Precision@10", f"{meta['precision_at_10']:.0%}",
              f"{meta['precision_at_10'] - meta['baseline_precision_at_10']:+.0%} vs form baseline")
    c3.metric("Winner in our top 5", f"{meta['winner_in_top5']:.0%}",
              f"{meta['winner_in_top5'] - meta['baseline_winner_in_top5']:+.0%} vs form baseline")
    c4.metric("Stage-type accuracy", f"{meta['stage_type_accuracy']:.0%}")

    left, right = st.columns(2)
    with left:
        imp = pd.Series(meta["feature_importance"]).sort_values()
        fig = px.bar(imp, orientation="h", title="Permutation importance (AUC drop)",
                     labels={"value": "AUC drop when shuffled", "index": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        preds = pd.read_csv(MODELS / "test_predictions.csv")
        preds["bin"] = pd.cut(preds["p_top10"], [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1])
        cal = preds.groupby("bin", observed=True).agg(predicted=("p_top10", "mean"),
                                                     actual=("top10", "mean"),
                                                     n=("top10", "size")).reset_index()
        fig = px.scatter(cal, x="predicted", y="actual", size="n", title="Calibration")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Points on the dashed line = predicted probabilities match reality.")

# ---------------------------------------------------------------- riders
with tab_rider:
    res = add_percentiles(results).merge(stages, on="stage_id")
    options = feats.sort_values("p_top10", ascending=False)["rider_id"]
    rid = st.selectbox("Rider", options, format_func=lambda r: f"{names[r]} ({teams[r]})")
    mine = res[res["rider_id"] == rid]

    c1, c2, c3 = st.columns(3)
    c1.metric("Starts", len(mine))
    c2.metric("Wins", int((mine["rank"] == 1).sum()))
    c3.metric("Top-10s", int(mine["top10"].sum()))

    left, right = st.columns(2)
    with left:
        prof = mine.groupby("stage_type").agg(mean_pct=("pct", "mean"),
                                              top10_rate=("top10", "mean")).reindex(STAGE_TYPES)
        fig = px.bar(prof.reset_index(), x="stage_type", y="top10_rate", color="stage_type",
                     color_discrete_map=TYPE_COLORS, title="Top-10 rate by terrain",
                     labels={"top10_rate": "Top-10 rate", "stage_type": ""})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(mine, x="date", y="rank", color="stage_type", log_y=True,
                         color_discrete_map=TYPE_COLORS, hover_data=["race"],
                         title="Every result (lower = better)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
