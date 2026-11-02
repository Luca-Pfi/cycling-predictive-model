"""Breakaway & winner predictions for the ANJ-listed pro races.

    streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cyclepredict import config, data, evaluate, features, insights, live, models, predict

st.set_page_config(page_title="Breakaway Predictor", page_icon="🚴", layout="wide")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29b"


@st.cache_resource(show_spinner="Building features for every stage…")
def load():
    t, rows, stages = models.prepare()
    hist = features.rider_history(t)
    return t, rows, stages, hist


@st.cache_resource
def load_predictor():
    return predict.load_predictor()


def odds(p):
    return f"{1 / max(p, 1e-4):.2f}"


def mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


if not (config.MODELS / "predictor.joblib").exists():
    st.error("No trained models. Run `python -m cyclepredict demo` (or fetch real data) and "
             "`python -m cyclepredict train`.")
    st.stop()

t, rows, stages, hist = load()
P = load_predictor()
today = data.today()
names = t["riders"].set_index("rider_id")["name"].to_dict()
name = lambda r: names.get(r, r)
cal = pd.read_csv(config.CALENDAR).set_index("pcs_slug")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    if config.is_demo():
        st.warning(f"**DEMO DATA**: fictional riders on the real calendar, demo clock {today.date()}. "
                   "Fetch real PCS data on your computer to switch (see the Data tab).")
    else:
        st.success(f"Real PCS data · today {today.date()}")
    races = t["races"].copy()
    races["label"] = races["slug"].map(cal["name"]).fillna(races["name"]) + " " + races["year"].astype(str)
    when = st.radio("Races", ["Upcoming", "Recent", "All"], horizontal=True)
    if when == "Upcoming":
        pick = races[(races["startdate"] >= today - pd.Timedelta(days=25)) & (races["startdate"] <= today + pd.Timedelta(days=60))]
        pick = pick[pick["race_id"].isin(stages.loc[~stages["finished"].fillna(False).astype(bool), "race_id"])]
    elif when == "Recent":
        pick = races[(races["startdate"] < today) & (races["startdate"] >= today - pd.Timedelta(days=120))]
    else:
        pick = races
    pick = pick.sort_values("startdate", ascending=when != "Recent")
    if pick.empty:
        st.info("No race in this window.")
        st.stop()
    race_id = st.selectbox("Race", pick["race_id"], format_func=lambda r: races.set_index("race_id").at[r, "label"])
    rs = stages[stages["race_id"] == race_id].sort_values("stage_index")
    rs = rs[rs["stage_id"].isin(rows["stage_id"])]
    if rs.empty:
        st.info("No startlist for this race yet: the preview appears as soon as PCS publishes it.")
        st.stop()
    open_st = rs[~rs["finished"].fillna(False).astype(bool)]
    default = rs.index.get_loc(open_st.index[0]) if len(open_st) else 0
    sid = st.selectbox("Stage", rs["stage_id"], index=int(default),
                       format_func=lambda s: f"{'Stage ' + s.rsplit('-', 1)[-1] if 'stage-' in s else 'Race'} · "
                                             f"{rs.set_index('stage_id').at[s, 'date'].date()}")

S = t["stages"].set_index("stage_id").loc[sid]
stage = S.to_dict()
stage["stage_id"] = sid
r, s = predict.preview(P, rows, stages, [sid])
s0 = s.iloc[0]
p_bw = float(s0["p_break_wins"])
r["name"] = r["rider_id"].map(name)
asof = min(S["date"], today)
prof = insights.rider_profiles(hist, asof)
pat = insights.race_patterns(t["stages"], S["slug"], asof, features.ICON_GROUP.get(S["profile_icon"]))
is_itt = S["stage_type"] == "ITT"

st.title(f"{cal['name'].get(S['slug'], S['slug'])} {S['year']}"
         + (f" · stage {S['stage_index']}" if S["n_stages"] > 1 else ""))
climbs = ", ".join(f"{c}×{k.replace('climbs_', '').upper()}" for k, c in
                   ((k, int(S[k])) for k in ["climbs_hc", "climbs_1", "climbs_2", "climbs_3", "climbs_4"]) if c)
st.caption(f"{S['date'].date()} · {S['distance_km']:.0f} km · {S['vertical_m']:.0f} m climbing · profile score "
           f"{S['profile_score']:.0f} ({S['profile_icon']}{', uphill finish' if S['profile_icon'] in ('p3', 'p5') else ''})"
           + (f" · climbs: {climbs}, last: {S['last_climb']}" if climbs else "") + f" · {len(r)} starters"
           + (" · **already raced**" if S["finished"] else ""))

tab_pre, tab_live, tab_track, tab_data = st.tabs(["🔮 Pre-race", "🔴 Live", "📊 Track record", "ℹ️ Data"])

# ------------------------------------------------------------------ pre-race
with tab_pre:
    c = st.columns(4)
    if is_itt:
        c[0].metric("Breakaway", "n/a (time trial)")
    else:
        c[0].metric("Breakaway wins", f"{100 * p_bw:.0f}%", f"fair odds yes {odds(p_bw)} · no {odds(1 - p_bw)}", delta_color="off")
        c[1].metric("Expected break size", f"{s0['expected_break_size']:.0f} riders")
    fav = r.nlargest(1, "p_win").iloc[0]
    c[2].metric("Favourite", fav["name"], f"{100 * fav['p_win']:.0f}% · fair odds {odds(fav['p_win'])}", delta_color="off")
    if pat["n"]:
        c[3].metric("Past stages here", f"{pat['n']}",
                    f"break won {100 * pat['break_win_rate']:.0f}%" if pat["break_win_rate"] == pat["break_win_rate"] else "",
                    delta_color="off")

    left, right = st.columns(2)
    with left:
        top = r.nlargest(15, "p_win").iloc[::-1]
        via_b = p_bw * top["p_break"] * top["p_win_if_break"]
        fig = go.Figure([
            go.Bar(y=top["name"], x=top["p_win"] - via_b, orientation="h", name="via the bunch", marker_color=BLUE),
            go.Bar(y=top["name"], x=via_b, orientation="h", name="via the break", marker_color=ORANGE,
                   text=[f"{odds(p)}" for p in top["p_win"]], textposition="outside")])
        fig.update_layout(barmode="stack", title="Win probability (label: fair odds)", height=520,
                          xaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.1), margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
    with right:
        if not is_itt:
            br = r.nlargest(15, "p_break").iloc[::-1]
            fig = go.Figure(go.Bar(y=br["name"], x=br["p_break"], orientation="h", marker_color=ORANGE,
                                   text=[odds(p) for p in br["p_break"]], textposition="outside",
                                   customdata=br["team"], hovertemplate="%{y} (%{customdata})<br>%{x:.0%}<extra></extra>"))
            fig.update_layout(title="Who goes in the break (label: fair odds)", height=520, xaxis_tickformat=".0%",
                              margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Key paths to victory")
    picks = list(r.nlargest(5, "p_win")["rider_id"]) + [x for x in r.nlargest(8, "p_break")["rider_id"] if x not in set(r.nlargest(5, "p_win")["rider_id"])][:5]
    cols = st.columns(2)
    for i, rid in enumerate(picks):
        row = r[r["rider_id"] == rid].iloc[0].to_dict()
        pr = prof.loc[rid].to_dict() if rid in prof.index else None
        with cols[i % 2].expander(f"{row['name']} · {pr['type'] if pr else 'rider'} · win {100 * row['p_win']:.1f}% "
                                  f"· break {100 * row['p_break']:.0f}%", expanded=i < 2):
            for tip in insights.key_paths(stage, row, pr, pat, p_bw):
                st.markdown(f"- {tip}")

    vers = predict.versions(sid)
    if len(vers) > 1:
        st.subheader("How the preview changed")
        hv = pd.DataFrame([{"made_at": m["made_at"], "P(break wins)": m["p_break_wins"], "starters": m["n_starters"]}
                           for m, _ in vers])
        fig = px.line(hv, x="made_at", y="P(break wins)", markers=True)
        fig.update_traces(line_color=BLUE)
        fig.update_layout(yaxis_tickformat=".0%", height=260)
        st.plotly_chart(fig, width="stretch")

    if S["finished"]:
        res = t["results"][t["results"]["stage_id"] == sid]
        w = res[res["rank"] == 1]
        brk = res[res["in_break"]]["rider_id"].map(name).tolist()
        st.info(f"**Result**: {name(w['rider_id'].iloc[0]) if len(w) else '?'} won ({S['won_how']}). "
                + (f"Break ({len(brk)}): {', '.join(brk)} — it {'won' if S.get('break_won') else 'was caught'}." if brk else ""))

# ------------------------------------------------------------------ live
with tab_live:
    if is_itt:
        st.info("Time trial: no breakaway. Use the pre-race win odds.")
    else:
        st.caption("Enter the race situation (from the PCS live ticker or TV). The chase model turns gap and distance "
                   "into the break's chance; every estimate you save becomes training data for the live model.")
        lm = live.LiveModel()
        lm.fit(stages)
        c1, c2, c3 = st.columns([2, 1, 1])
        default_break = r.nlargest(int(round(s0["expected_break_size"])), "p_break")["rider_id"].tolist()
        brk = c1.multiselect("Riders in the break", r.sort_values("p_break", ascending=False)["rider_id"],
                             default=default_break, format_func=name)
        km = c2.slider("Km to go", 0.0, float(S["distance_km"]), float(min(60, S["distance_km"])), 0.5)
        gap = c3.text_input("Gap (m:ss)", "2:30")
        try:
            mm, ss = gap.split(":") if ":" in gap else (0, gap)
            gap_s = int(mm) * 60 + int(ss)
        except ValueError:
            gap_s = 150
            st.warning("Gap format: m:ss")
        c4, c5 = st.columns(2)
        chase = c4.select_slider("Chase", list(live.CHASE), value="several teams")
        hard = c5.checkbox("Climb / hard finish still to come", S["profile_icon"] in ("p3", "p5"))
        state = live.LiveState(stage_id=sid, km_to_go=km, gap_s=gap_s, break_riders=brk, chase=chase, hard_finish=hard)
        p_s = lm.p_survive(state, p_bw, S["distance_km"])
        k = st.columns(4)
        k[0].metric("Break survives", f"{100 * p_s:.0f}%", f"fair odds {odds(p_s)}", delta_color="off")
        k[1].metric("Break caught", f"{100 * (1 - p_s):.0f}%", f"fair odds {odds(1 - p_s)}", delta_color="off")
        k[2].metric("Pre-race call", f"{100 * p_bw:.0f}%")
        k[3].metric("Model", "learned from logs" if lm.fitted is not None else "chase rule-of-thumb")
        need = lm.gap_needed(state, km_points=tuple(x for x in (100, 80, 60, 50, 40, 30, 20, 15, 10, 5, 2) if x <= S["distance_km"]))
        left, right = st.columns([1, 1])
        with left:
            fig = go.Figure([go.Scatter(x=list(need), y=[v / 60 for v in need.values()], mode="lines+markers",
                                        name="gap needed (50/50)", line=dict(color=GREY, dash="dash")),
                             go.Scatter(x=[km], y=[gap_s / 60], mode="markers", name="now",
                                        marker=dict(color=ORANGE, size=14))])
            fig.update_layout(title="Gap the break needs", xaxis=dict(title="km to go", autorange="reversed"),
                              yaxis_title="minutes", height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        with right:
            lo = live.rider_odds(P, rows, S, state, p_s)
            lo["name"] = lo["rider_id"].map(name)
            show = lo.head(25)
            st.dataframe(show[["name", "team", "in_break", "p_win", "fair_odds", "p_win_if_survives"]], hide_index=True,
                         width="stretch", height=380,
                         column_config={"name": "Rider", "team": "Team", "in_break": "In break",
                                        "p_win": st.column_config.ProgressColumn("P(win)", format="percent", min_value=0, max_value=1),
                                        "fair_odds": st.column_config.NumberColumn("Fair odds", format="%.2f"),
                                        "p_win_if_survives": st.column_config.NumberColumn("P(win | break survives)", format="percent")})
        st.subheader("Key paths, live")
        cols = st.columns(2)
        for i, rid in enumerate(list(brk) + list(lo[~lo["in_break"]].head(3)["rider_id"])):
            row = r[r["rider_id"] == rid].iloc[0].to_dict()
            pr = prof.loc[rid].to_dict() if rid in prof.index else None
            p_rider = float(lo.loc[lo["rider_id"] == rid, "p_win"].iloc[0])
            with cols[i % 2].expander(f"{row['name']} · {'in the break' if rid in brk else 'chasing'} · "
                                      f"{100 * p_rider:.1f}% · odds {odds(p_rider)}"):
                for tip in insights.key_paths(stage, row, pr, pat, p_bw, live=lm, break_riders=brk if rid in brk else None,
                                              p_survive=p_s if rid in brk else None):
                    st.markdown(f"- {tip}")
        if st.button("Save this live estimate"):
            live.log_snapshot(state, p_s)
            st.success("Saved to data/live_log; it will be scored and learned from once the stage is over.")

# ------------------------------------------------------------------ track record
with tab_track:
    meta = json.loads((config.MODELS / "metrics.json").read_text())
    o = meta["overall"]
    st.caption(f"Walk-forward test: each season predicted by models trained only on earlier seasons "
               f"({', '.join(map(str, meta['test_years']))}). Trained {meta['trained_at']} on {meta['data']} data.")
    k = st.columns(5)
    k[0].metric("Break wins: AUC", f"{o['bw_auc']:.2f}", f"race-history baseline {o['bw_auc_baseline']:.2f}", delta_color="off")
    k[1].metric("Break wins: Brier", f"{o['bw_brier']:.3f}", f"baseline {o['bw_brier_baseline']:.3f}", delta_color="off")
    k[2].metric("Break riders: AUC", f"{o['member_auc']:.2f}", f"rider break-rate baseline {o['member_auc_baseline']:.2f}", delta_color="off")
    k[3].metric("Break riders: hits", f"{100 * o['member_precision_at_k']:.0f}%", "of the top-k picks were in it", delta_color="off")
    k[4].metric("Winner in our top 3", f"{100 * o['winner_top3']:.0f}%", f"form baseline {100 * o['winner_top3_baseline']:.0f}%", delta_color="off")
    track = pd.read_csv(evaluate.TRACK) if evaluate.TRACK.exists() else pd.DataFrame()
    if len(track):
        off = track[track["official"]]
        left, right = st.columns(2)
        with left:
            lab = off.dropna(subset=["break_won"])
            lab = lab.assign(bin=pd.cut(lab["p_break_wins"], [0, .05, .1, .2, .3, .4, .5, .6, .8, 1]))
            calib = lab.groupby("bin", observed=True).agg(pred=("p_break_wins", "mean"), obs=("break_won", "mean"),
                                                         n=("break_won", "size")).reset_index()
            fig = px.scatter(calib, x="pred", y="obs", size="n", title="Is P(break wins) calibrated?",
                             labels={"pred": "predicted", "obs": "observed"})
            fig.update_traces(marker_color=BLUE)
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=GREY))
            fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=380)
            st.plotly_chart(fig, width="stretch")
        with right:
            summ = evaluate.summary(track)
            summ["race"] = summ["slug"].map(cal["name"]).fillna(summ["slug"])
            st.markdown("**Accuracy per race** (official pre-race previews)")
            st.dataframe(summ[["race", "stages", "bw_brier", "break_hits", "winner_top3", "p_winner"]]
                         .sort_values("bw_brier", ascending=False), hide_index=True, width="stretch", height=340,
                         column_config={"race": "Race", "stages": "Stages", "bw_brier": st.column_config.NumberColumn("Break Brier ↓", format="%.3f"),
                                        "break_hits": st.column_config.NumberColumn("Break picks hit", format="percent"),
                                        "winner_top3": st.column_config.NumberColumn("Winner in top 3", format="percent"),
                                        "p_winner": st.column_config.NumberColumn("P given to winner", format="percent")})
        miss = off[off["note"].notna()].sort_values("date", ascending=False)
        st.markdown(f"**Biggest misses** ({len(miss)} of {len(off)} stages)")
        st.dataframe(miss.assign(race=miss["slug"].map(cal["name"]))[["date", "race", "stage_id", "p_break_wins", "note", "backtest"]],
                     hide_index=True, width="stretch", height=260)
    else:
        st.info("No scored previews yet: run `python -m cyclepredict evaluate` after races.")

# ------------------------------------------------------------------ data
with tab_data:
    st.markdown(f"""
**Source:** `{config.data_dir()}` ({'DEMO — fictional riders' if config.is_demo() else 'ProCyclingStats via the procyclingstats package'})
· {len(t['races'])} race editions · {int(stages['finished'].fillna(False).sum())} raced stages · {len(t['riders'])} riders.

**Breakaway definition:** a rider is in the break when PCS records ≥ max({config.BREAK_MIN_KM:.0f} km,
{100 * config.BREAK_MIN_SHARE:.0f}% of the distance) ahead of the peloton; the break *wins* when the winner was in it.
Races where PCS has no breakaway km are used for the winner models only.

**Real data (on your computer):**
```
python -m cyclepredict check
python -m cyclepredict verify-calendar
python -m cyclepredict fetch --years 2019-2025
python -m cyclepredict train
python -m cyclepredict update      # each race day: results, startlists, scoring, retraining, previews
```
""")
    cov = t["results"].merge(t["stages"][["stage_id", "year"]], on="stage_id").groupby("year")["breakaway_kms"].apply(lambda x: (x > 0).mean())
    st.markdown("**Share of result rows with breakaway km, per season**")
    st.dataframe(cov.rename("share").to_frame().T, width="stretch")
