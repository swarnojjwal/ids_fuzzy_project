
"""
Fuzzy Logic Intrusion Detection System - Streamlit dashboard
Run with:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import skfuzzy as fuzz
import streamlit as st
from skfuzzy import control as ctrl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

st.set_page_config(page_title="Fuzzy IDS", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------
# Data columns (NSL-KDD)
# ----------------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

COLORS = {"Normal": "#2ecc71", "Suspicious": "#f39c12", "Intrusion": "#e74c3c"}


# ----------------------------------------------------------------------------
# Fuzzy system (cached so it is built only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def build_fuzzy_system():
    count = ctrl.Antecedent(np.arange(0, 512, 1), "count")
    failed = ctrl.Antecedent(np.arange(0, 6, 1), "failed")
    serror = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "serror")
    risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

    count["low"] = fuzz.trimf(count.universe, [0, 0, 100])
    count["medium"] = fuzz.trimf(count.universe, [50, 150, 300])
    count["high"] = fuzz.trimf(count.universe, [200, 511, 511])

    failed["low"] = fuzz.trimf(failed.universe, [0, 0, 2])
    failed["medium"] = fuzz.trimf(failed.universe, [1, 2.5, 4])
    failed["high"] = fuzz.trimf(failed.universe, [3, 5, 5])

    serror["low"] = fuzz.trimf(serror.universe, [0, 0, 0.4])
    serror["medium"] = fuzz.trimf(serror.universe, [0.2, 0.5, 0.8])
    serror["high"] = fuzz.trimf(serror.universe, [0.6, 1, 1])

    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk["medium"] = fuzz.trimf(risk.universe, [25, 50, 75])
    risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(failed["high"], risk["high"]),
        ctrl.Rule(serror["high"], risk["high"]),
        ctrl.Rule(count["high"] & serror["medium"], risk["medium"]),
        ctrl.Rule(count["medium"] & serror["medium"], risk["medium"]),
        ctrl.Rule(count["low"] & failed["low"] & serror["low"], risk["low"]),
        ctrl.Rule(count["medium"] & failed["low"] & serror["low"], risk["low"]),
        ctrl.Rule(count["high"] & serror["low"], risk["medium"]),
        ctrl.Rule(failed["medium"] & serror["medium"], risk["medium"]),
        ctrl.Rule(count["low"], risk["low"]),
        ctrl.Rule(count["medium"], risk["medium"]),
        ctrl.Rule(count["high"], risk["medium"]),
    ]
    rule_text = [
        "IF failed logins is HIGH THEN risk is HIGH",
        "IF SYN error rate is HIGH THEN risk is HIGH",
        "IF count is HIGH AND SYN error is MEDIUM THEN risk is MEDIUM",
        "IF count is MEDIUM AND SYN error is MEDIUM THEN risk is MEDIUM",
        "IF count is LOW AND failed is LOW AND SYN error is LOW THEN risk is LOW",
        "IF count is MEDIUM AND failed is LOW AND SYN error is LOW THEN risk is LOW",
        "IF count is HIGH AND SYN error is LOW THEN risk is MEDIUM",
        "IF failed is MEDIUM AND SYN error is MEDIUM THEN risk is MEDIUM",
        "IF count is LOW THEN risk is LOW  (fallback)",
        "IF count is MEDIUM THEN risk is MEDIUM  (fallback)",
        "IF count is HIGH THEN risk is MEDIUM  (fallback)",
    ]
    system = ctrl.ControlSystem(rules)
    return system, {"count": count, "failed": failed, "serror": serror, "risk": risk}, rule_text


fuzzy_system, variables, rule_text = build_fuzzy_system()


def classify(score, normal_max, suspicious_max):
    if score < normal_max:
        return "Normal"
    if score < suspicious_max:
        return "Suspicious"
    return "Intrusion"


def compute_risk(sim, count, failed, serror):
    sim.input["count"] = float(np.clip(count, 0, 511))
    sim.input["failed"] = float(np.clip(failed, 0, 5))
    sim.input["serror"] = float(np.clip(serror, 0, 1))
    sim.compute()
    return round(float(sim.output["risk"]), 2)


def new_sim():
    return ctrl.ControlSystemSimulation(fuzzy_system)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def gauge(score, normal_max, suspicious_max):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, normal_max], "color": COLORS["Normal"]},
                    {"range": [normal_max, suspicious_max], "color": COLORS["Suspicious"]},
                    {"range": [suspicious_max, 100], "color": COLORS["Intrusion"]},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def membership_plot(var, title, marker=None):
    fig = go.Figure()
    for term in var.terms:
        fig.add_trace(
            go.Scatter(x=var.universe, y=var[term].mf, mode="lines", name=term,
                       fill="tozeroy", opacity=0.6)
        )
    if marker is not None:
        fig.add_vline(x=marker, line_dash="dash", line_color="black")
    fig.update_layout(title=title, height=260, margin=dict(t=40, b=20, l=10, r=10),
                      yaxis_title="Membership", legend=dict(orientation="h"))
    return fig


@st.cache_data(show_spinner=False)
def load_data(file_bytes_or_path):
    return pd.read_csv(file_bytes_or_path, header=None, names=COLUMNS)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🛡️ Fuzzy IDS")
st.sidebar.caption("Intrusion detection with fuzzy logic on NSL-KDD")
st.sidebar.subheader("Decision thresholds")
normal_max = st.sidebar.slider("Normal if risk <", 10, 60, 40)
suspicious_max = st.sidebar.slider("Intrusion if risk ≥", 50, 95, 70)
if suspicious_max <= normal_max:
    st.sidebar.error("Intrusion threshold must be higher than the Normal threshold.")
    st.stop()

st.title("🛡️ Network Intrusion Detection System")
st.caption("Fuzzy-logic risk scoring using connection count, failed logins and SYN error rate.")

tab_live, tab_batch, tab_fuzzy = st.tabs(
    ["🔍 Live Detector", "📊 Batch Analysis", "🧠 Fuzzy System"]
)

# ----------------------------------------------------------------------------
# TAB 1: live detector
# ----------------------------------------------------------------------------
with tab_live:
    st.subheader("Analyse a single connection")

    presets = {
        "Custom": None,
        "Normal traffic": (20, 0, 0.0),
        "Busy but healthy": (180, 0, 0.1),
        "Possible port scan": (350, 1, 0.5),
        "SYN flood (DoS)": (450, 0, 0.95),
        "Brute-force login": (80, 5, 0.1),
    }
    preset = st.selectbox("Scenario preset", list(presets))
    p = presets[preset] or (100, 0, 0.2)

    c1, c2, c3 = st.columns(3)
    count_in = c1.slider("Connection count", 0, 511, p[0])
    failed_in = c2.slider("Failed logins", 0, 5, p[1])
    serror_in = c3.slider("SYN error rate", 0.0, 1.0, float(p[2]), 0.01)

    score = compute_risk(new_sim(), count_in, failed_in, serror_in)
    verdict = classify(score, normal_max, suspicious_max)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(score, normal_max, suspicious_max), width="stretch")
    with right:
        icon = {"Normal": "✅", "Suspicious": "⚠️", "Intrusion": "🚨"}[verdict]
        msg = f"{icon} **{verdict.upper()}**  —  risk score {score}"
        {"Normal": st.success, "Suspicious": st.warning, "Intrusion": st.error}[verdict](msg)

        st.markdown("**How strongly each input belongs to each fuzzy set**")
        rows = []
        for name, val in [("count", count_in), ("failed", failed_in), ("serror", serror_in)]:
            var = variables[name]
            row = {"Input": name, "Value": val}
            for term in ["low", "medium", "high"]:
                row[term] = round(float(fuzz.interp_membership(var.universe, var[term].mf, val)), 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.plotly_chart(membership_plot(variables["count"], "Connection count", count_in), width="stretch")
    m2.plotly_chart(membership_plot(variables["failed"], "Failed logins", failed_in), width="stretch")
    m3.plotly_chart(membership_plot(variables["serror"], "SYN error rate", serror_in), width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: batch analysis
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Evaluate on the NSL-KDD test set")

    uploaded = st.file_uploader("Upload KDDTest+.txt (or any NSL-KDD file)", type=["txt", "csv"])
    source = uploaded
    if source is None and os.path.exists("KDDTest+.txt"):
        source = "KDDTest+.txt"
        st.info("Using KDDTest+.txt found in the app folder.")

    if source is None:
        st.warning("Upload a dataset (or place KDDTest+.txt next to app.py) to run batch detection.")
    else:
        df = load_data(source)
        df["actual"] = np.where(df["label"].str.lower() == "normal", "Normal", "Intrusion")

        c1, c2 = st.columns(2)
        n = c1.slider("Sample size", 100, min(len(df), 5000), min(1000, len(df)), step=100)
        seed = c2.number_input("Random seed", value=42, step=1)

        if st.button("▶ Run detection", type="primary"):
            sample = df.sample(n=n, random_state=int(seed)).copy()
            sim = new_sim()
            scores = []
            bar = st.progress(0.0, text="Scoring connections...")
            for i, (_, r) in enumerate(sample.iterrows()):
                scores.append(compute_risk(sim, r["count"], r["num_failed_logins"], r["serror_rate"]))
                if i % 50 == 0:
                    bar.progress((i + 1) / n, text=f"Scoring connections... {i + 1}/{n}")
            bar.empty()
            sample["risk_score"] = scores
            st.session_state["results"] = sample

        if "results" in st.session_state:
            res = st.session_state["results"].copy()
            res["prediction"] = [classify(s, normal_max, suspicious_max) for s in res["risk_score"]]
            res["binary_pred"] = np.where(res["prediction"] == "Normal", "Normal", "Intrusion")

            acc = accuracy_score(res["actual"], res["binary_pred"])
            prec, rec, f1, _ = precision_recall_fscore_support(
                res["actual"], res["binary_pred"], pos_label="Intrusion",
                average="binary", zero_division=0,
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Records", len(res))
            k2.metric("Accuracy", f"{acc * 100:.2f}%")
            k3.metric("Precision", f"{prec * 100:.2f}%")
            k4.metric("Recall (detection rate)", f"{rec * 100:.2f}%")
            k5.metric("F1-score", f"{f1 * 100:.2f}%")

            a, b = st.columns(2)
            with a:
                dist = res["prediction"].value_counts().reindex(list(COLORS), fill_value=0).reset_index()
                dist.columns = ["Class", "Records"]
                fig = px.bar(dist, x="Class", y="Records", color="Class",
                             color_discrete_map=COLORS, title="Predicted classes")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")
            with b:
                cm = confusion_matrix(res["actual"], res["binary_pred"], labels=["Normal", "Intrusion"])
                fig = px.imshow(cm, text_auto=True, x=["Normal", "Intrusion"], y=["Normal", "Intrusion"],
                                color_continuous_scale="Blues", title="Confusion matrix",
                                labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig, width="stretch")

            fig = px.histogram(res, x="risk_score", color="actual", nbins=40, barmode="overlay",
                               opacity=0.7, title="Risk score distribution by true class",
                               color_discrete_map={"Normal": COLORS["Normal"], "Intrusion": COLORS["Intrusion"]})
            fig.add_vline(x=normal_max, line_dash="dash")
            fig.add_vline(x=suspicious_max, line_dash="dash")
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Detected connections")
            pick = st.multiselect("Filter by prediction", list(COLORS), default=["Suspicious", "Intrusion"])
            view = res[res["prediction"].isin(pick)][
                ["count", "num_failed_logins", "serror_rate", "protocol_type", "service",
                 "label", "actual", "risk_score", "prediction"]
            ].sort_values("risk_score", ascending=False)
            st.dataframe(view, width="stretch", hide_index=True)
            st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode(),
                               "ids_results.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 3: fuzzy system
# ----------------------------------------------------------------------------
with tab_fuzzy:
    st.subheader("Membership functions")
    c1, c2 = st.columns(2)
    c1.plotly_chart(membership_plot(variables["count"], "Connection count"), width="stretch")
    c2.plotly_chart(membership_plot(variables["failed"], "Failed logins"), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(membership_plot(variables["serror"], "SYN error rate"), width="stretch")
    c4.plotly_chart(membership_plot(variables["risk"], "Risk (output)"), width="stretch")

    st.subheader("Rule base")
    st.table(pd.DataFrame({"#": range(1, len(rule_text) + 1), "Rule": rule_text}).set_index("#"))