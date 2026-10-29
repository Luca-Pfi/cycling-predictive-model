"""Page 2: career trajectories and talent scouting (penalised linear regression)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.career_model import CUTOFFS, LABELS, MAX_AGE, cutoff_features, project, summarise

MODELS = Path("models/career")
BLUE, BLUE_BAND, ORANGE, GREY = "#2a78d6", "rgba(42,120,214,0.16)", "#eb6834", "#a3a29b"
LEVEL_NAMES = {"amateur": "Amateur", "conti": "Continental", "pro": "ProTeam", "wt": "WorldTeam"}


@st.cache_resource
def load_artifacts():
    summary = json.loads((MODELS / "summary.json").read_text())
    arts = {int(c): joblib.load(MODELS / f"cutoff_{c}.joblib") for c in summary["cutoffs"]}
    return summary, arts


@st.cache_data
def load_careers(data_dir):
    d = Path(data_dir)
    return pd.read_csv(d / "riders.csv"), pd.read_csv(d / "seasons.csv")


@st.cache_data
def features_at(data_dir, cutoff):
    riders, seasons = load_careers(data_dir)
    return cutoff_features(riders, seasons, cutoff)


@st.cache_data
def prospect_board(_arts, data_dir):
    """Every rider aged 21-25 who raced last season, projected from today."""
    riders, seasons = load_careers(data_dir)
    last = int(seasons["season"].max())
    now = seasons[seasons["season"] == last]
    boards, curves = [], []
    for c, art in _arts.items():
        ids = set(now.loc[now["age"] == c, "rider_id"])
        f = features_at(data_dir, c)
        f = f[f["rider_id"].isin(ids)]
        if f.empty:
            continue
        cur = project(art, f)
        summ = summarise(cur)
        p30 = cur[cur["target_age"] == max(30, c + 1)].set_index("rider_id")["p_active"]
        summ["p_racing_30"] = summ["rider_id"].map(p30)
        summ = summ.merge(f[["rider_id", "lp_now", "current_level", "wins_total", "u23_wins_total"]],
                          on="rider_id")
        summ["age"] = c
        boards.append(summ)
        curves.append(cur[["rider_id", "target_age", "mu_pts", "lo_pts", "hi_pts", "p_active"]])
    info = riders.reindex(columns=["rider_id", "name", "nationality", "specialty"]).fillna({"nationality": "–", "specialty": "–"})
    board = pd.concat(boards).merge(info, on="rider_id")
    board["points_now"] = np.expm1(board["lp_now"]).round()
    return board.sort_values("peak_pts", ascending=False), pd.concat(curves), last


def similar_riders(art, feats, row, data_dir, k=5):
    """Nearest historical riders at the same cutoff, in the model's standardised space."""
    riders, seasons = load_careers(data_dir)
    last = int(seasons["season"].max())
    hist = feats[(feats["birth_year"] + 31 <= last) & (feats["rider_id"] != row["rider_id"].iloc[0])]
    hist = hist[~hist["rider_id"].isin(art["test_riders"])] if row["rider_id"].iloc[0] in art["test_riders"] else hist
    ct = art["model"]["design"].ct_
    Xh, x = ct.transform(hist), ct.transform(row)
    d = np.sqrt(((Xh - x) ** 2).sum(axis=1))
    near = hist.iloc[np.argsort(d)[:k]]["rider_id"]
    traj = seasons[seasons["rider_id"].isin(near)].merge(riders[["rider_id", "name"]], on="rider_id")
    return near.tolist(), traj


def career_chart(curve, actual, cutoff, similar=None, title=""):
    fig = go.Figure()
    if similar is not None:
        for i, (rid, g) in enumerate(similar.groupby("rider_id")):
            fig.add_trace(go.Scatter(
                x=g["age"], y=g["points"].clip(lower=1), mode="lines", name="Similar riders at the same age",
                legendgroup="sim", showlegend=i == 0, line=dict(color=GREY, width=1),
                opacity=0.55, hovertemplate=f"{g['name'].iloc[0]}<br>age %{{x}}: %{{y:,.0f}} pts<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(curve["target_age"]) + list(curve["target_age"])[::-1],
                             y=list(curve["hi_pts"].clip(lower=1)) + list(curve["lo_pts"].clip(lower=1))[::-1],
                             fill="toself", fillcolor=BLUE_BAND, line=dict(width=0), hoverinfo="skip",
                             name="80% prediction interval"))
    fig.add_trace(go.Scatter(
        x=curve["target_age"], y=curve["mu_pts"].clip(lower=1), mode="lines", name="Projection",
        line=dict(color=BLUE, width=2.5),
        customdata=np.c_[curve["lo_pts"], curve["hi_pts"], curve["p_active"]],
        hovertemplate="age %{x}: %{y:,.0f} pts<br>80%: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
                      "<br>P(still racing) %{customdata[2]:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=actual["age"], y=actual["points"].clip(lower=1), mode="lines+markers",
                             name="Actual", line=dict(color=ORANGE, width=2), marker=dict(size=8),
                             hovertemplate="age %{x}: %{y:,.0f} pts<extra></extra>"))
    fig.add_vline(x=cutoff + 0.5, line=dict(color=GREY, dash="dot"))
    fig.add_annotation(x=cutoff + 0.5, y=1, yref="paper", text="  prediction starts", showarrow=False,
                       xanchor="left", font=dict(size=11, color=GREY))
    fig.update_layout(title=title, height=460, hovermode="closest", margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(title="Age", range=[18.5, MAX_AGE + 0.5], dtick=2),
                      yaxis=dict(title="PCS points per season (log scale)", type="log"))
    return fig


def survival_chart(curve, height=240):
    fig = px.line(curve, x="target_age", y="p_active", markers=True,
                  labels={"target_age": "Age", "p_active": "P(still racing)"},
                  title="Probability of still racing")
    fig.update_traces(line_color=BLUE, line_width=2,
                      hovertemplate="age %{x}: %{y:.0%}<extra></extra>")
    fig.update_layout(height=height, yaxis=dict(range=[0, 1.02], tickformat=".0%"),
                      margin=dict(t=50, b=10))
    return fig


# ------------------------------------------------------------------------ page
if not (MODELS / "summary.json").exists():
    st.error("No career models found. Run `python -m src.make_career_data` and "
             "`python -m src.train_career` first.")
    st.stop()

summary, arts = load_artifacts()
data_dir = summary["data_dir"]
riders, seasons = load_careers(data_dir)
names = riders.set_index("rider_id")["name"]
board, prospect_curves, last_season = prospect_board(arts, data_dir)

st.title("📈 Career trajectory & talent scouting")
st.caption(f"Data: `{data_dir}` · {len(riders):,} riders · {len(seasons):,} rider-seasons · "
           f"projections from the end of the {last_season} season")
with st.expander("How the model works"):
    m = arts[23]["metrics"]
    st.markdown(f"""
A rider's file is frozen at a **cutoff age** (21–25). Everything known by then is used to
predict their **log PCS points at every later age**:

* **{m['n_features_raw']} raw variables**: lab tests (VO2max, FTP, sprint power), body (height,
  weight, BMI), background and age they started cycling, nationality, specialty, junior results,
  U23 wins/podiums, Tour de l'Avenir and U23 Worlds, points in each season so far, trend, wins,
  top-10s, race days, Grand Tours, team level and changes, injuries...
* **{m['n_design_columns']} model columns** after imputation (+ missing-value flags), one-hot
  encoding, a **cubic spline of the target age** (the average career curve) and **spline × variable
  interactions** (so variables can bend the shape of the curve, not just move it up or down).
* **OLS, Ridge, Lasso and Elastic Net** are compared with 5-fold cross-validation **grouped by
  rider** (no rider is in both training and validation). Best at age 23: **{m['best_model']}**.
* **Survivorship**: a logistic regression estimates P(still racing) at each age; weak riders
  retire, so their missing seasons are not random.
* **Prediction intervals**: split-conformal, from out-of-fold residuals at each horizon.
* **Inference**: a separate OLS on main effects with rider-clustered standard errors gives
  coefficients, 95% CIs and p-values, after removing variables with VIF > 10.
""")

tab_stars, tab_rider, tab_whatif, tab_model = st.tabs(
    ["🌟 Future stars", "📈 Rider projection", "🎛️ What-if simulator", "🔬 Model insight"])

# ---------------------------------------------------------------- future stars
with tab_stars:
    c1, c2 = st.columns([2, 1])
    specs = c1.multiselect("Specialty", sorted(board["specialty"].unique()),
                           default=sorted(board["specialty"].unique()))
    ages = c2.multiselect("Age now", sorted(board["age"].unique()), default=sorted(board["age"].unique()))
    view = board[board["specialty"].isin(specs) & board["age"].isin(ages)]

    top = view.iloc[0] if len(view) else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prospects (age 21–25)", len(view))
    if top is not None:
        k2.metric("Highest projected peak", top["name"], f"{top['peak_pts']:,.0f} pts at {top['peak_age']}",
                  delta_color="off")
    k3.metric("Projected 1,000+ pt seasons", int((view["peak_pts"] >= 1000).sum()))
    k4.metric("Median projected peak", f"{view['peak_pts'].median():,.0f} pts")

    fig = px.scatter(view, x="peak_age", y="peak_pts", log_y=True, hover_name="name",
                     hover_data={"age": True, "specialty": True, "nationality": True,
                                 "points_now": ":,.0f", "peak_pts": ":,.0f", "peak_age": True},
                     labels={"peak_age": "Projected peak age", "peak_pts": "Projected peak-season points",
                             "points_now": f"Points in {last_season}", "age": "Age now"},
                     title="Projected peak: how good, and when")
    fig.update_traces(marker=dict(color=BLUE, size=9, opacity=0.75, line=dict(color="white", width=1)))
    if top is not None:
        fig.add_annotation(x=top["peak_age"], y=np.log10(top["peak_pts"]), text=top["name"], showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=12))
    fig.update_layout(height=430, xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

    show = view.head(30).assign(
        level=lambda d: d["current_level"].map(LEVEL_NAMES),
        interval=lambda d: [f"{a:,.0f} – {b:,.0f}" for a, b in zip(d["peak_lo"], d["peak_hi"])])
    st.dataframe(
        show[["name", "age", "nationality", "specialty", "level", "points_now", "u23_wins_total",
              "peak_pts", "interval", "peak_age", "p_racing_30", "expected_career_pts"]],
        column_config={
            "name": "Rider", "age": "Age", "nationality": "Nat.", "specialty": "Specialty",
            "level": "Team level", "points_now": st.column_config.NumberColumn(f"Pts {last_season}", format="%d"),
            "u23_wins_total": "U23 wins",
            "peak_pts": st.column_config.NumberColumn("Projected peak", format="%d"),
            "interval": "80% interval", "peak_age": "Peak age",
            "p_racing_30": st.column_config.ProgressColumn("P(racing at 30)", format="percent", min_value=0, max_value=1),
            "expected_career_pts": st.column_config.NumberColumn("Expected points, rest of career", format="%d"),
        }, hide_index=True, width="stretch")

# ------------------------------------------------------------ rider projection
with tab_rider:
    mode = st.radio("Riders", ["Current prospects", "Backtest: held-out riders from the past"],
                    horizontal=True,
                    help="Backtest riders were never seen by the model: their whole career was held out, "
                         "so you can compare the projection with what really happened.")
    if mode == "Current prospects":
        rid = st.selectbox("Rider", board["rider_id"],
                           format_func=lambda r: f"{names[r]} · age {board.set_index('rider_id').loc[r, 'age']}")
        cutoff = int(board.set_index("rider_id").loc[rid, "age"])
        holdout = False
    else:
        c1, c2 = st.columns([1, 3])
        cutoff = c1.selectbox("Cutoff age", list(arts), index=list(arts).index(23) if 23 in arts else 0)
        art = arts[cutoff]
        feats_c = features_at(data_dir, cutoff)
        test = feats_c[feats_c["rider_id"].isin(art["test_riders"]) &
                       (feats_c["birth_year"] + cutoff + 4 <= last_season)]
        peak = seasons.groupby("rider_id")["points"].max()
        test = test.assign(peak=test["rider_id"].map(peak)).sort_values("peak", ascending=False)
        rid = c2.selectbox("Held-out rider (sorted by real career peak)", test["rider_id"],
                           format_func=lambda r: f"{names[r]} · real peak {peak[r]:,.0f} pts")
        holdout = True

    art = arts[cutoff]
    feats_c = features_at(data_dir, cutoff)
    row = feats_c[feats_c["rider_id"] == rid]
    curve = project(art, row, holdout=holdout)
    near, sim_traj = similar_riders(art, feats_c, row, data_dir)
    actual = seasons[seasons["rider_id"] == rid]
    s = summarise(curve).iloc[0]
    info = riders.set_index("rider_id").loc[rid]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected peak season", f"{s['peak_pts']:,.0f} pts",
              f"80%: {s['peak_lo']:,.0f}–{s['peak_hi']:,.0f}", delta_color="off")
    k2.metric("Projected peak age", int(s["peak_age"]))
    k3.metric("Expected points after cutoff", f"{s['expected_career_pts']:,.0f}")
    if holdout:
        future = actual[actual["age"] > cutoff]
        k4.metric("Real peak after cutoff", f"{future['points'].max():,.0f} pts" if len(future) else "retired")
    else:
        p30 = curve.loc[curve["target_age"] == max(30, cutoff + 1), "p_active"].iloc[0]
        k4.metric("P(still racing at 30)", f"{p30:.0%}")

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(career_chart(curve, actual, cutoff, sim_traj,
                                     " · ".join(str(info.get(k, "")) for k in ["name", "specialty", "nationality"] if pd.notna(info.get(k)))),
                        width="stretch")
    with right:
        st.markdown("**Rider file at the cutoff**")
        r0 = row.iloc[0]
        facts = {"Age at cutoff": str(cutoff), "Team level": LEVEL_NAMES.get(r0.get("current_level"), "–"),
                 f"Points at {cutoff}": f"{np.expm1(r0['lp_now']):,.0f}"}
        for key, label, fmt in [("vo2max", "VO2max", "{:.1f}"), ("ftp_wkg_u23", "FTP at 19 (W/kg)", "{:.2f}"),
                                ("sprint_peak_w", "Sprint (W)", "{:.0f}"), ("height_cm", "Height (cm)", "{:.0f}"),
                                ("weight_kg", "Weight (kg)", "{:.0f}"), ("started_cycling_age", "Started cycling at", "{:.0f}"),
                                ("u23_wins_total", "U23 wins", "{:.0f}"), ("wins_total", "Pro wins so far", "{:.0f}"),
                                ("injuries", "Injury seasons", "{:.0f}")]:
            if pd.notna(r0.get(key)):
                facts[label] = fmt.format(r0[key])
        st.dataframe(pd.DataFrame(facts.items(), columns=["", "value"]),
                     hide_index=True, width="stretch")
    st.plotly_chart(survival_chart(curve), width="stretch")
    st.caption("Grey lines: the 5 historical riders whose file at the same age was most similar "
               "(distance in the model's standardised feature space) and what really happened to them.")

# ---------------------------------------------------------------- what-if
with tab_whatif:
    st.markdown("Change a prospect's file and watch the linear model redraw the career curve.")
    rid_w = st.selectbox("Prospect", board["rider_id"], key="whatif_rider",
                         format_func=lambda r: f"{names[r]} · age {board.set_index('rider_id').loc[r, 'age']}")
    cw = int(board.set_index("rider_id").loc[rid_w, "age"])
    base = features_at(data_dir, cw).query("rider_id == @rid_w").copy()
    b0 = base.iloc[0]
    c1, c2, c3 = st.columns(3)
    mod = base.copy()
    has = lambda col: col in b0 and pd.notna(b0[col])
    pts = c1.slider(f"Points in the cutoff season ({last_season})", 0, 3000, int(np.expm1(b0["lp_now"])), step=10)
    # (column, widget column, label, min, max, step, transform from/to the slider value)
    knobs = [("vo2max", c1, "VO2max", 60.0, 95.0, 0.5, float, float),
             ("ftp_wkg_u23", c1, "FTP at 19 (W/kg)", 4.0, 6.8, 0.05, float, float),
             ("u23_wins_total", c2, "U23 wins", 0, 30, 1, int, int),
             ("avenir_score", c2, "Best Tour de l'Avenir rank (31 = not top 30)", 1, 31, 1,
              lambda v: int(31 - v), lambda v: 31 - v),
             ("started_cycling_age", c2, "Age started cycling", 6, 22, 1, int, int),
             ("wt_seasons", c3, "Seasons in a WorldTeam", 0, cw - 18, 1, int, int)]
    for col, box, label, lo, hi, step, to_slider, from_slider in knobs:
        if has(col):
            mod[col] = from_slider(box.slider(label, lo, hi, to_slider(b0[col]), step))
    if has("current_level"):
        levels = list(LEVEL_NAMES)
        mod["current_level"] = c3.selectbox("Team level now", levels, index=levels.index(b0["current_level"]),
                                            format_func=LEVEL_NAMES.get)
    if has("injured_now"):
        mod["injured_now"] = int(c3.checkbox("Injured in the cutoff season", bool(b0["injured_now"])))

    lp_new = np.log1p(pts)
    mod["lp_now"] = lp_new
    # best season so far: if it was the cutoff season, fall back to the seasons before it
    earlier_best = b0["lp_max"] if b0["lp_max"] > b0["lp_now"] else np.nanmax([b0["lp_prev"], b0["lp_prev2"], 0])
    mod["lp_max"] = max(earlier_best, lp_new)
    hist = [v for v in [b0["lp_prev2"], b0["lp_prev"]] if pd.notna(v)] + [lp_new]
    mod["lp_slope"] = np.polyfit(range(len(hist)), hist, 1)[0] if len(hist) > 1 else np.nan
    mod["lp_mean"] = (b0["lp_mean"] * b0["seasons_observed"] - b0["lp_now"] + lp_new) / b0["seasons_observed"]
    mod["career_points_log"] = np.log1p(max(np.expm1(b0["career_points_log"]) - np.expm1(b0["lp_now"]) + pts, 0))

    orig, new = project(arts[cw], base), project(arts[cw], mod)
    so, sn = summarise(orig).iloc[0], summarise(new).iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Projected peak", f"{sn['peak_pts']:,.0f} pts", f"{sn['peak_pts'] - so['peak_pts']:+,.0f} vs original")
    k2.metric("Peak age", int(sn["peak_age"]), f"{int(sn['peak_age'] - so['peak_age']):+d} years", delta_color="off")
    k3.metric("Expected points, rest of career", f"{sn['expected_career_pts']:,.0f}",
              f"{sn['expected_career_pts'] - so['expected_career_pts']:+,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(new["target_age"]) + list(new["target_age"])[::-1],
                             y=list(new["hi_pts"].clip(lower=1)) + list(new["lo_pts"].clip(lower=1))[::-1],
                             fill="toself", fillcolor="rgba(235,104,52,0.14)", line=dict(width=0),
                             hoverinfo="skip", name="What-if 80% interval"))
    for curve_, name, color, dash, shift in [(orig, "Original file", BLUE, "dash", -9),
                                             (new, "What-if", ORANGE, "solid", 9)]:
        fig.add_trace(go.Scatter(x=curve_["target_age"], y=curve_["mu_pts"].clip(lower=1), name=name,
                                 mode="lines", line=dict(color=color, width=2.5, dash=dash),
                                 hovertemplate="age %{x}: %{y:,.0f} pts<extra>" + name + "</extra>"))
        end = curve_.iloc[-1]
        fig.add_annotation(x=end["target_age"], y=np.log10(max(end["mu_pts"], 1)), text=name, xanchor="left",
                           showarrow=False, xshift=6, yshift=shift, font=dict(size=12))
    past = seasons[seasons["rider_id"] == rid_w]
    fig.add_trace(go.Scatter(x=past["age"], y=past["points"].clip(lower=1), mode="markers", name="Actual so far",
                             marker=dict(color=GREY, size=8)))
    fig.update_layout(height=430, yaxis=dict(type="log", title="PCS points per season (log scale)"),
                      xaxis=dict(title="Age", range=[18.5, MAX_AGE + 2.5], dtick=2),
                      legend=dict(orientation="h", y=-0.18), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- model insight
with tab_model:
    cm = st.selectbox("Cutoff age", list(arts), index=list(arts).index(23) if 23 in arts else 0, key="insight_cut")
    art = arts[cm]
    m = art["metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Test R² (log points)", f"{m['test_r2']:.3f}")
    k2.metric("Test RMSE (log points)", f"{m['test_rmse']:.3f}")
    k3.metric("80% interval coverage", f"{m['interval_coverage_80']:.1%}")
    k4.metric("Survival model AUC", f"{m['survival_auc']:.3f}")
    k5.metric("Peak-season rank corr.", f"{m['peak_spearman']:.3f}",
              help=f"Spearman correlation between predicted and real best future season, {m['peak_n']} test riders")

    st.subheader("Model comparison")
    comp = art["comparison"].copy()
    comp["params"] = comp["params"].astype(str).replace("{}", "–")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(comp.sort_values("cv_rmse"), hide_index=True, width="stretch",
                     column_config={"model": "Model", "params": "Best hyper-parameters",
                                    "cv_rmse": st.column_config.NumberColumn("CV RMSE", format="%.3f"),
                                    "cv_r2": st.column_config.NumberColumn("CV R²", format="%.3f"),
                                    "test_rmse": st.column_config.NumberColumn("Test RMSE", format="%.3f"),
                                    "test_r2": st.column_config.NumberColumn("Test R²", format="%.3f"),
                                    "n_columns": "Columns", "n_nonzero": "Non-zero coefs"})
    with right:
        fig = px.bar(comp.sort_values("test_r2"), x="test_r2", y="model", orientation="h",
                     labels={"test_r2": "Test R² (held-out riders)", "model": ""}, text_auto=".3f")
        fig.update_traces(marker_color=[ORANGE if n == m["best_model"] else BLUE for n in comp.sort_values("test_r2")["model"]])
        fig.update_layout(height=300, margin=dict(t=10), xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Orange = selected by grouped CV ({m['best_model']}).")

    st.subheader("What drives a career? (interpretable OLS, standardised coefficients)")
    coef = art["coef_table"].assign(t=lambda d: d["coef"] / d["se"])
    top = coef.reindex(coef["t"].abs().sort_values(ascending=False).index).head(22).iloc[::-1]
    sig = top["p_value"] < 0.05
    fig = go.Figure()
    for flag, color, name in [(True, BLUE, "p < 0.05"), (False, GREY, "not significant")]:
        d = top[sig == flag]
        fig.add_trace(go.Scatter(
            x=d["coef"], y=d["label"], mode="markers", name=name, marker=dict(color=color, size=9),
            error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["coef"],
                         arrayminus=d["coef"] - d["ci_low"], color=color, thickness=1.5),
            customdata=np.c_[d["ci_low"], d["ci_high"], d["p_value"]],
            hovertemplate="%{y}<br>coef %{x:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                          "<br>p = %{customdata[2]:.2g}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=GREY, width=1))
    fig.update_layout(height=620, xaxis_title="Effect on future log points per +1 SD (95% CI, rider-clustered)",
                      yaxis=dict(categoryorder="array", categoryarray=list(top["label"])),
                      legend=dict(orientation="h", y=-0.08), margin=dict(t=10))
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, width="stretch")
    with right:
        st.markdown(f"OLS R² = **{m['inference_ols_r2']:.3f}** (adjusted {m['inference_ols_r2_adj']:.3f}). "
                    "Each coefficient is the change in future log points for a 1-standard-deviation "
                    "increase, holding the others fixed. Categorical effects are relative to the first "
                    "category.")
        st.markdown("**Multicollinearity (VIF)**: variables above 10 were removed from this "
                    "inference model; the predictive Lasso / Ridge keep them all because the penalty "
                    "copes with collinearity.")
        vif = art["vif"].copy()
        vif["vif"] = vif["vif"].map(lambda v: "∞ (exact)" if v >= 1e6 else f"{v:.1f}")
        st.dataframe(vif[["label", "vif", "in_inference_model"]], hide_index=True, width="stretch", height=330,
                     column_config={"label": "Variable", "vif": "VIF", "in_inference_model": "Kept"})

    st.subheader("Penalised model: strongest terms")
    pc = art["penalised_coefs"].assign(abs=lambda d: d["coef"].abs())
    nz = (pc["abs"] > 1e-8).sum()
    st.caption(f"{nz} of {len(pc)} columns are non-zero in the final {m['best_model']}. "
               "Terms with ' x age_spline_k' are interactions that change the *shape* of the career curve.")
    st.dataframe(pc.sort_values("abs", ascending=False).head(20)[["term", "coef"]],
                 hide_index=True, width="stretch",
                 column_config={"coef": st.column_config.NumberColumn("Coefficient", format="%.3f")})

    st.subheader("Diagnostics on held-out riders")
    dg = art["diagnostics"]
    a, b, c = st.columns(3)
    fig = px.scatter(dg, x="pred", y="target_lp", opacity=0.25,
                     labels={"pred": "Predicted log points", "target_lp": "Actual log points"},
                     title="Predicted vs actual")
    fig.update_traces(marker=dict(color=BLUE, size=5))
    lim = [0, max(dg["pred"].max(), dg["target_lp"].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False))
    a.plotly_chart(fig, width="stretch")
    fig = px.box(dg, x="horizon", y="resid", labels={"horizon": "Years after cutoff", "resid": "Residual"},
                 title="Residuals by horizon")
    fig.update_traces(marker_color=BLUE, line_color=BLUE)
    b.plotly_chart(fig, width="stretch")
    (osm, osr), (slope, inter, _) = stats.probplot(dg["resid"], dist="norm")
    fig = go.Figure([go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=BLUE, size=4), name="residuals"),
                     go.Scatter(x=osm, y=slope * osm + inter, mode="lines", line=dict(color=GREY, dash="dash"),
                                name="normal")])
    fig.update_layout(title="Normal Q-Q plot", xaxis_title="Theoretical quantiles",
                      yaxis_title="Residual quantiles", showlegend=False)
    c.plotly_chart(fig, width="stretch")

    st.subheader("Does waiting a year help?")
    rows = [dict(cutoff=c, test_r2=a_["metrics"]["test_r2"], coverage=a_["metrics"]["interval_coverage_80"],
                 width=float(np.mean(np.array(a_["q_hi"][:5]) - np.array(a_["q_lo"][:5]))),
                 peak_spearman=a_["metrics"]["peak_spearman"]) for c, a_ in arts.items()]
    cut = pd.DataFrame(rows)
    a, b = st.columns(2)
    fig = px.line(cut, x="cutoff", y="peak_spearman", markers=True, title="Ranking accuracy of the career peak",
                  labels={"cutoff": "Cutoff age", "peak_spearman": "Spearman (pred vs real peak)"})
    fig.update_traces(line_color=BLUE, marker_size=9)
    fig.update_layout(yaxis_range=[0, 1], xaxis_dtick=1)
    a.plotly_chart(fig, width="stretch")
    fig = px.line(cut, x="cutoff", y="width", markers=True, title="Width of the 80% interval (next 5 seasons)",
                  labels={"cutoff": "Cutoff age", "width": "Interval width (log points)"})
    fig.update_traces(line_color=BLUE, marker_size=9)
    fig.update_layout(yaxis_rangemode="tozero", xaxis_dtick=1)
    b.plotly_chart(fig, width="stretch")
