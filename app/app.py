"""Streamlit front-end: Dengue Early-Warning for Sri Lankan districts.

Run locally:  streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import load_bundle, load_history, predict_week, risk_band

st.set_page_config(page_title="Dengue Early Warning – Sri Lanka", page_icon="🦟", layout="wide")

BAND_COLORS = {"Very high": "#9b1d2a", "High": "#d64545", "Elevated": "#e9a23b", "Low": "#5b8c5a"}
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.risk-card {border-radius: 14px; padding: 18px 22px; color: white; margin-bottom: 8px;}
.risk-card h2 {margin: 0; font-size: 2.2rem; color: white;}
.risk-card p {margin: 2px 0 0 0; opacity: .92;}
.small-note {color: #6b7280; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_history():
    return load_history()


bundle, hist = get_bundle(), get_history()
THR = bundle["decision_threshold"]
dates = sorted(hist["date"].unique(), reverse=True)
min_date = hist["date"].min() + pd.Timedelta(weeks=60)
dates = [d for d in dates if d >= min_date]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🦟 Settings")
    report_date = st.selectbox("Report week (week starting)", dates, index=0,
                               format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    target_date = pd.Timestamp(report_date) + pd.Timedelta(weeks=bundle["horizon_weeks"])
    st.caption(f"Forecast target: week of **{target_date:%d %b %Y}** ({bundle['horizon_weeks']} weeks ahead)")
    st.divider()
    st.subheader("Model")
    st.write(f"**{bundle['model_name']}**")
    tm = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Test PR-AUC", f"{tm.get('PR-AUC', tm.get('pr_auc', 0)):.3f}")
    c2.metric("Test F1", f"{tm.get('F1', tm.get('f1', 0)):.3f}")
    c1.metric("Recall", f"{tm.get('Recall', tm.get('recall', 0)):.3f}")
    c2.metric("ROC-AUC", f"{tm.get('ROC-AUC', tm.get('roc_auc', 0)):.3f}")
    st.caption(f"Decision threshold: {THR:.2f} · trained on {bundle['trained_on']}")
    st.divider()
    st.caption("Data: Epidemiology Unit, Ministry of Health Sri Lanka – weekly dengue notifications "
               "(via the `denguedatahub` package, T. S. Talagala, USJ). For education and research. "
               "Not an official alert system.")

# ------------------------------------------------------------------ header
st.title("Dengue Early Warning — Sri Lanka")
st.markdown("Will a district have a **high-transmission dengue week** two weeks from now? "
            "*High transmission* means more cases than the district's 75th percentile for 2012–2021.")

tab_nat, tab_dist, tab_ho, tab_perf = st.tabs(["🗺️ National overview", "🔍 District forecast & what-if",
                                                "✅ Hold-out check", "📊 Model performance"])

# ------------------------------------------------------------------ national overview
with tab_nat:
    pred = predict_week(bundle, hist, report_date)
    pred["band"] = [risk_band(p, THR) for p in pred["probability"]]
    known = pred["actual_target_cases"].notna().all()
    n_high = int(pred["high_risk_pred"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts forecast high-risk", f"{n_high} / 26")
    c2.metric("Cases reported this week (all districts)", f"{int(pred['cases'].sum()):,}")
    c3.metric("Districts already above threshold", int((pred["rel_now"] > 0).sum()))
    if known:
        acc = (pred["high_risk_pred"] == (pred["actual_target_cases"] > pred["threshold"]).astype(int)).mean()
        c4.metric("Forecast accuracy for this week (actual known)", f"{acc:.0%}")
    else:
        c4.metric("Outcome", "Not yet known")

    chart = alt.Chart(pred).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("probability:Q", title="P(high-transmission week)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("district:N", sort="-x", title=None),
        color=alt.Color("band:N", scale=alt.Scale(domain=list(BAND_COLORS), range=list(BAND_COLORS.values())), title="Risk"),
        tooltip=["district", "province", alt.Tooltip("probability:Q", format=".2f"), "cases", alt.Tooltip("threshold:Q", format=".0f")],
    ).properties(height=560)
    rule = alt.Chart(pd.DataFrame({"x": [THR]})).mark_rule(strokeDash=[4, 4], color="black").encode(x="x:Q")
    left, right = st.columns([3, 2])
    left.altair_chart(chart + rule, width="stretch")
    show = pred[["district", "province", "cases", "threshold", "probability", "band", "actual_target_cases"]].rename(columns={
        "cases": "cases this week", "threshold": "threshold", "probability": "P(high risk)",
        "band": "risk", "actual_target_cases": "actual cases (target wk)"})
    right.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "P(high risk)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "threshold": st.column_config.NumberColumn(format="%.0f"),
        "actual cases (target wk)": st.column_config.NumberColumn(format="%.0f")})

# ------------------------------------------------------------------ district forecast
with tab_dist:
    districts = sorted(hist["district"].unique())
    district = st.selectbox("District (RDHS area)", districts, index=districts.index("Colombo"))
    h = hist[(hist["district"] == district) & (hist["date"] <= report_date)].tail(8)[["date", "cases"]].copy()
    thr_cases = bundle["reference"]["threshold"][district]

    st.markdown("**Recent weekly cases.** Edit any value to run a what-if scenario, or to type in this week's count before it's in the dataset.")
    colA, colB = st.columns([2, 3])
    with colA:
        edit = st.data_editor(h.assign(date=h["date"].dt.strftime("%Y-%m-%d")).rename(columns={"date": "week starting"}),
                              hide_index=True, disabled=["week starting"], width="stretch",
                              column_config={"cases": st.column_config.NumberColumn(min_value=0, step=1)})
        overrides = {district: dict(zip(edit["week starting"], edit["cases"]))}
        changed = not np.allclose(edit["cases"].values, h["cases"].values)
    base = predict_week(bundle, hist, report_date)
    base_row = base[base["district"] == district].iloc[0]
    res = predict_week(bundle, hist, report_date, overrides if changed else None)
    row = res[res["district"] == district].iloc[0]
    p = float(row["probability"]); band = risk_band(p, THR)

    with colB:
        st.markdown(f"""<div class="risk-card" style="background:{BAND_COLORS[band]}">
            <p>{district} · forecast for week of {row['target_date']:%d %b %Y}</p>
            <h2>{band} risk — {p:.0%}</h2>
            <p>Probability that cases will exceed <b>{thr_cases:.0f}</b> (the district threshold). Alert raised when it's {THR:.0%} or higher.</p>
            </div>""", unsafe_allow_html=True)
        if changed:
            st.caption(f"What-if scenario. With the original data the probability was {base_row['probability']:.0%}.")
        if not np.isnan(row["actual_target_cases"]) and not changed:
            outcome = "HIGH" if row["actual_target_cases"] > thr_cases else "normal"
            st.info(f"Actual outcome: **{row['actual_target_cases']:.0f} cases**, a {outcome} week.")

        hh = hist[(hist["district"] == district) & (hist["date"] <= target_date) & (hist["date"] > pd.Timestamp(report_date) - pd.Timedelta(weeks=30))].copy()
        hh["period"] = np.where(hh["date"] <= report_date, "known at report week", "future (to be forecast)")
        ed = edit.assign(date=pd.to_datetime(edit["week starting"]))
        hh = hh.merge(ed[["date", "cases"]].rename(columns={"cases": "edited"}), on="date", how="left")
        hh["cases"] = np.where(hh["edited"].notna(), hh["edited"], hh["cases"])
        bars = alt.Chart(hh).mark_bar().encode(
            x=alt.X("date:T", title=None), y=alt.Y("cases:Q", title="weekly cases"),
            color=alt.Color("period:N", scale=alt.Scale(range=["#5c7080", "#d9dee3"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["date:T", "cases:Q"])
        thr_line = alt.Chart(pd.DataFrame({"y": [thr_cases]})).mark_rule(color="#d64545", strokeDash=[5, 3]).encode(y="y:Q")
        tgt = alt.Chart(pd.DataFrame({"x": [row["target_date"]]})).mark_rule(color="black").encode(x="x:T")
        st.altair_chart((bars + thr_line + tgt).properties(height=260), width="stretch")

    with st.expander("Why this prediction? (exact GAM explanation)", expanded=True):
        model = bundle["model"]
        x_row = res.loc[res["district"] == district, bundle["features"]]
        if hasattr(model, "term_contributions"):  # spline GAM: the prediction IS the sum of these terms
            intercept = model.intercept_
            contrib = model.term_contributions(x_row).iloc[0]
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
            cdf = pd.DataFrame({"term": top.index, "contribution": top.values})
            cdf["direction"] = np.where(cdf["contribution"] > 0, "raises risk", "lowers risk")
            ch = alt.Chart(cdf).mark_bar().encode(
                x=alt.X("contribution:Q", title="contribution to log-odds of a high-risk week"),
                y=alt.Y("term:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#d64545", "#2a6f97"]), title=None),
                tooltip=["term", alt.Tooltip("contribution:Q", format="+.3f")]).properties(height=320)
            st.altair_chart(ch, width="stretch")
            logit = intercept + contrib.sum()
            st.caption(f"The spline GAM is a glass-box model: baseline {intercept:+.2f} + the sum of all {len(contrib)} term contributions "
                       f"({contrib.sum():+.2f}) = {logit:+.2f} log-odds, which is P = {1 / (1 + np.exp(-logit)):.2f}. Terms with '&' are smooth pairwise (tensor-product) interactions. "
                       "Feature values are log ratios to the district threshold: 0 means 'at threshold' and −1 means about 2.7× below it.")
        else:
            st.info("Per-prediction explanation is available for the spline GAM model.")
        desc = pd.DataFrame({"feature": bundle["features"], "value": row[bundle["features"]].astype(float).round(3).values,
                             "meaning": [bundle["feature_descriptions"][f] for f in bundle["features"]]})
        st.dataframe(desc, hide_index=True, width="stretch")

# ------------------------------------------------------------------ hold-out check
with tab_ho:
    st.markdown("One data point was **held out of training, validation and testing**: the latest Colombo week with a known outcome. "
                "The model reloaded from `dengue_model.pkl` predicts it here.")
    ho = pd.read_csv(ROOT / "data" / "holdout_sample.csv", parse_dates=["date", "target_date"])
    r = ho.iloc[0]
    p_ho = bundle["model"].predict_proba(ho[bundle["features"]])[:, 1][0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Report week", f"{r['date']:%d %b %Y}", f"{int(r['cases'])} cases that week", delta_color="off", delta_arrow="off")
    c2.metric("Predicted", f"{'HIGH RISK' if p_ho >= THR else 'normal'}", f"P = {p_ho:.2f}", delta_color="off", delta_arrow="off")
    c3.metric(f"Actual (week of {r['target_date']:%d %b})", f"{'HIGH RISK' if r['high_risk'] == 1 else 'normal'}",
              f"{int(r['target_cases'])} cases vs threshold {r['threshold']:.0f}", delta_color="off", delta_arrow="off")
    st.success("Correct prediction ✔" if (p_ho >= THR) == (r["high_risk"] == 1) else "Incorrect prediction ✘")
    st.dataframe(ho[["district", "date", "target_date", "cases", "threshold"] + bundle["features"]].assign(
                     date=lambda x: x["date"].dt.strftime("%Y-%m-%d"), target_date=lambda x: x["target_date"].dt.strftime("%Y-%m-%d"))
                 .round(3).T.rename(columns={0: "value"}).astype(str),
                 width="stretch")

# ------------------------------------------------------------------ performance
with tab_perf:
    res_path = ROOT / "reports" / "test_results.csv"
    if res_path.exists():
        st.subheader("Test set (Jan 2024 – May 2026), never used for training or tuning")
        st.dataframe(pd.read_csv(res_path, index_col=0).round(3), width="stretch")
    figs = [("test_roc_pr.png", "ROC and precision–recall curves"), ("test_confusion_calibration.png", "Confusion matrix and calibration"),
            ("gam_importance.png", "GAM global term importance"), ("gam_shape_functions.png", "GAM shape functions with 95% intervals"), ("test_per_district.png", "Per-district performance"),
            ("test_timelines.png", "Forecasts vs actual outcomes")]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        fp = ROOT / "reports" / "figures" / f
        if fp.exists():
            cols[i % 2].image(str(fp), caption=cap, width="stretch")
