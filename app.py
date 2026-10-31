"""
Fuzzy Logic Intrusion Detection System (Mamdani + Sugeno): 3 inputs, 27 rules, NSL-KDD.
Run with:  streamlit run app.py        (all visual design is in styles.py)
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import styles as ui
from styles import CLASSES, CYAN, GREEN, MODEL_COLORS, PURPLE, RED, show, style

# =============================================================================
# 1. DATA FILES AND ATTACK FAMILIES
# =============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "KDDTrain_.txt")
TEST_PATH = os.path.join(BASE, "data", "KDDTest_.txt")

# Only the 3 fuzzy inputs and the label are read (column positions in the NSL-KDD files).
USECOLS = [22, 24, 32, 41]
COLNAMES = ["count", "serror_rate", "dst_host_srv_count", "label"]

FAMILY = {}
for _n in "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split():
    FAMILY[_n] = "DoS"
for _n in "ipsweep nmap portsweep satan mscan saint".split():
    FAMILY[_n] = "Probe"
for _n in "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm".split():
    FAMILY[_n] = "U2R"
for _n in ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster "
           "snmpguess snmpgetattack named xlock xsnoop sendmail").split():
    FAMILY[_n] = "R2L"

# =============================================================================
# 2. THE 3 FUZZY INPUTS AND THEIR MEMBERSHIP FUNCTIONS (scaled to 0..1 by x / cap)
# =============================================================================
VARIABLES = {
    "count": dict(label="Connection count", cap=511,
                  info="Connections to the same host in the recent time window."),
    "serror_rate": dict(label="SYN error rate", cap=1.0,
                        info="Share of connections with SYN errors (failed handshakes, floods)."),
    "dst_host_srv_count": dict(label="Destination-host service count", cap=255,
                               info="Connections to the same service on the destination host."),
}
MFS = {
    "count": {"low": ("trap", [0.0, 0.0, 0.03, 0.15]),
              "medium": ("tri", [0.08, 0.30, 0.58]),
              "high": ("trap", [0.42, 0.70, 1.0, 1.0])},
    "serror_rate": {"low": ("trap", [0.0, 0.0, 0.05, 0.30]),
                    "medium": ("tri", [0.15, 0.50, 0.85]),
                    "high": ("trap", [0.60, 0.85, 1.0, 1.0])},
    "dst_host_srv_count": {"low": ("trap", [0.0, 0.0, 0.10, 0.30]),
                           "medium": ("tri", [0.15, 0.45, 0.80]),
                           "high": ("trap", [0.60, 0.90, 1.0, 1.0])},
}
# Mamdani output sets (risk score 0..100)
OUT_MF = {"Low": ("trap", [0, 0, 15, 40]), "Medium": ("tri", [25, 50, 75]), "High": ("trap", [60, 85, 100, 100])}
UNIVERSE = np.arange(0, 101, 1.0)


# =============================================================================
# 3. THE 27 FUZZY RULES  (3 x 3 x 3 grid: count x SYN error x service count)
# =============================================================================
def build_rules():
    terms = ("low", "medium", "high")
    rules = []
    for c in terms:
        for s in terms:
            for z in terms:
                score = {"low": 8, "medium": 45, "high": 88}[s]
                score += {"low": 0, "medium": 10, "high": 18}[c]
                score += {"low": 15, "medium": 5, "high": -8}[z]
                if c == "high" and s == "high":
                    score += 12
                if c == "high" and z == "low":
                    score += 8
                if s == "low" and z == "high":
                    score -= 8
                score = int(np.clip(score, 0, 100))
                rules.append(dict(
                    id=f"R{len(rules) + 1:02d}", cond=[("count", c), ("serror_rate", s), ("dst_host_srv_count", z)],
                    mam="Low" if score < 35 else ("Medium" if score < 70 else "High"), sug=float(score),
                    why=f"Connection count {c}, SYN error rate {s}, service count {z}: risk {score}/100."))
    return rules


RULES = build_rules()


def rule_text(r):
    return " AND ".join(f"{VARIABLES[v]['label']} is {s.upper()}" for v, s in r["cond"])


# =============================================================================
# 4. FUZZY LOGIC: membership, preprocessing, Mamdani and Sugeno inference
# =============================================================================
def membership(kind, p, x):
    x = np.asarray(x, dtype=float)
    if kind == "tri":
        a, b, c = p
        left = (x - a) / (b - a) if b > a else (x >= b).astype(float)
        right = (c - x) / (c - b) if c > b else (x <= b).astype(float)
        return np.clip(np.minimum(left, right), 0, 1)
    a, b, c, d = p                                              # trapezoid
    left = (x - a) / (b - a) if b > a else np.ones_like(x)
    right = (d - x) / (d - c) if d > c else np.ones_like(x)
    return np.clip(np.minimum(np.minimum(left, 1), right), 0, 1)


def preprocess(df, medians):
    """Raw values -> numbers; missing/invalid -> training median; clip to [0, cap]; scale to 0..1."""
    out = pd.DataFrame(index=df.index)
    for v, cfg in VARIABLES.items():
        raw = pd.to_numeric(df[v], errors="coerce") if v in df.columns else pd.Series(np.nan, index=df.index)
        raw = raw.mask(raw.isna() | np.isinf(raw) | (raw < 0), medians[v])
        out[v] = raw.clip(upper=cfg["cap"]) / cfg["cap"]
    return out


def firing(xn, how):
    """Rule strengths (N x 27). Mamdani AND = min, Sugeno AND = product."""
    mu = {(v, s): membership(k, p, xn[v].values) for v, sets in MFS.items() for s, (k, p) in sets.items()}
    cols = []
    for r in RULES:
        f = mu[r["cond"][0]]
        for c in r["cond"][1:]:
            f = np.minimum(f, mu[c]) if how == "mamdani" else f * mu[c]
        cols.append(f)
    return np.column_stack(cols)


def mamdani_infer(xn, return_agg=False):
    """min implication -> max aggregation -> centroid."""
    fire = firing(xn, "mamdani")
    agg = np.zeros((len(xn), len(UNIVERSE)))
    for label, (k, p) in OUT_MF.items():
        idx = [i for i, r in enumerate(RULES) if r["mam"] == label]
        strength = fire[:, idx].max(axis=1)
        agg = np.maximum(agg, np.minimum(strength[:, None], membership(k, p, UNIVERSE)[None, :]))
    area = agg.sum(axis=1)
    risk = np.where(area > 1e-9, (agg * UNIVERSE).sum(axis=1) / np.maximum(area, 1e-9), 50.0)
    return (risk, fire, agg) if return_agg else (risk, fire)


def sugeno_infer(xn):
    """Zero-order Sugeno: weighted average of the rule constants."""
    fire = firing(xn, "sugeno")
    z = np.array([r["sug"] for r in RULES])
    total = fire.sum(axis=1)
    return np.clip(np.where(total > 1e-9, (fire * z).sum(axis=1) / np.maximum(total, 1e-9), 50.0), 0, 100), fire


def run_models(df, med):
    xn = preprocess(df, med)
    m, m_fire = mamdani_infer(xn)
    s, s_fire = sugeno_infer(xn)
    return pd.DataFrame({"mamdani_risk": m, "sugeno_risk": s}, index=df.index), m_fire, s_fire


def classify(score, t_normal, t_malicious):
    return np.where(score < t_normal, "Normal", np.where(score < t_malicious, "Suspicious", "Malicious"))


# =============================================================================
# 5. DATA AND EVALUATION (no data leakage: borders tuned on a validation part of the training file)
# =============================================================================
def load_dataset(path):
    df = pd.read_csv(path, header=None, usecols=USECOLS, names=COLNAMES)
    name = df["label"].str.strip().str.lower()
    df["actual"] = np.where(name == "normal", "Normal", "Attack")
    df["family"] = name.map(lambda n: "Normal" if n == "normal" else FAMILY.get(n, "Other"))
    return df


@st.cache_data(show_spinner=False)
def get_data():
    train, test = load_dataset(TRAIN_PATH), load_dataset(TEST_PATH)
    return train, test, {v: float(train[v].median()) for v in VARIABLES}


def validation_split(train, seed=42, fit_fraction=0.7):
    order = np.random.RandomState(seed).permutation(len(train))
    return train.iloc[order[int(fit_fraction * len(train)):]]


def confusion(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return dict(TP=int((y & pred).sum()), TN=int((~y & ~pred).sum()),
                FP=int((~y & pred).sum()), FN=int((y & ~pred).sum()))


def metrics_from(c):
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(Accuracy=div(tp + tn, tp + tn + fp + fn), Precision=prec, Recall=rec,
                F1=div(2 * prec * rec, prec + rec), Specificity=div(tn, tn + fp),
                FPR=div(fp, fp + tn), FNR=div(fn, fn + tp))


def roc_curve(y, score):
    y = np.asarray(y).astype(int)
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], np.asarray(score)[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tpr, fpr = np.r_[0, tps[keep] / max(tps[-1], 1)], np.r_[0, fps[keep] / max(fps[-1], 1)]
    return fpr, tpr, float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def tune_thresholds(score, y):
    """Class borders from validation data only: best balance of recall and specificity, then the
    lowest risk above which at least 97 % of flagged records are real attacks."""
    y = np.asarray(y).astype(bool)
    best_t, best = 50.0, -1
    for t in np.arange(1, 100, 1.0):
        p = score >= t
        bal = ((p & y).sum() / max(y.sum(), 1) + (~p & ~y).sum() / max((~y).sum(), 1)) / 2
        if bal > best:
            best, best_t = bal, t
    t_mal = None
    for t in np.arange(best_t + 5, 100, 1.0):
        p = score >= t
        if p.sum() >= 50 and y[p].mean() >= 0.97:
            t_mal = t
            break
    return float(best_t), float(t_mal if t_mal is not None else best_t + (100 - best_t) / 2)


@st.cache_data(show_spinner="Running both fuzzy systems on the dataset ...")
def evaluate():
    train, test, med = get_data()
    val = validation_split(train)
    v_res, _, _ = run_models(val, med)
    y_val = (val["actual"] == "Attack").values
    thr = {m: tune_thresholds(v_res[m + "_risk"].values, y_val) for m in ("mamdani", "sugeno")}
    t_res, m_fire, s_fire = run_models(test, med)
    xn = preprocess(test.sample(5000, random_state=1), med)             # timing: median of 3 runs
    times = {"Mamdani": [], "Sugeno": []}
    for _ in range(3):
        a = time.perf_counter(); mamdani_infer(xn)
        b = time.perf_counter(); sugeno_infer(xn)
        c = time.perf_counter()
        times["Mamdani"].append(b - a); times["Sugeno"].append(c - b)
    ms = {k: float(np.median(v)) / len(xn) * 1e6 for k, v in times.items()}
    return dict(thr=thr, test=t_res, m_fire=m_fire, s_fire=s_fire, ms_per_1000=ms)


def compute_models(ev):
    """Metrics of both models on the TEST file (Suspicious + Malicious count as an alert)."""
    _, test, _ = get_data()
    y = (test["actual"] == "Attack").values
    out = {}
    for name, key in (("Mamdani", "mamdani"), ("Sugeno", "sugeno")):
        risk = ev["test"][key + "_risk"].values
        cls = classify(risk, *ev["thr"][key])
        fpr, tpr, auc = roc_curve(y, risk)
        c = confusion(y, cls != "Normal")
        out[name] = dict(m=metrics_from(c), c=c, fpr=fpr, tpr=tpr, auc=auc, cls=cls)
    return out


def verdict_text(M):
    a, b = M["Mamdani"], M["Sugeno"]
    pick = lambda k, hi=True: "Mamdani" if ((a[k] > b[k]) == hi) else "Sugeno"
    return [
        f"<b>Overall (F1-score):</b> {pick('F1')} is better (Mamdani {a['F1']:.3f} vs Sugeno {b['F1']:.3f}). "
        f"Accuracy: Mamdani {a['Accuracy']:.3f} vs Sugeno {b['Accuracy']:.3f}.",
        f"<b>Attack detection (recall):</b> {pick('Recall')} detects more attacks "
        f"(Mamdani {a['Recall']:.3f} vs Sugeno {b['Recall']:.3f}).",
        f"<b>False alarms:</b> {pick('FPR', False)} raises fewer (FPR Mamdani {a['FPR']:.3f} vs Sugeno {b['FPR']:.3f}).",
        f"<b>Ranking quality (AUC):</b> {pick('AUC')} is better (Mamdani {a['AUC']:.3f} vs Sugeno {b['AUC']:.3f}).",
        f"<b>Speed:</b> {pick('ms', False)} is faster (Mamdani {a['ms']:.1f} ms vs Sugeno {b['ms']:.1f} ms per 1000 records).",
    ]


# =============================================================================
# 6. SCORING OF MANUAL / UPLOADED / SIMULATED RECORDS
# =============================================================================
def score_records(df, source, ev):
    _, _, med = get_data()
    res, m_fire, s_fire = run_models(df, med)
    out = pd.DataFrame(index=df.index)
    out["source"] = source
    out["actual"] = df["actual"].values if "actual" in df.columns else "unknown"
    out["family"] = df["family"].values if "family" in df.columns else "unknown"
    out["mamdani_risk"], out["sugeno_risk"] = res["mamdani_risk"].round(2), res["sugeno_risk"].round(2)
    out["mamdani_class"] = classify(res["mamdani_risk"].values, *ev["thr"]["mamdani"])
    out["sugeno_class"] = classify(res["sugeno_risk"].values, *ev["thr"]["sugeno"])
    out["agree"] = out["mamdani_class"] == out["sugeno_class"]
    return out.reset_index(drop=True), m_fire, s_fire


def append_history(frame):
    frame = frame.copy()
    frame["batch"] = st.session_state["batch_no"] = st.session_state.get("batch_no", 0) + 1
    frame["time"] = pd.Timestamp.now().strftime("%H:%M:%S")
    start = st.session_state.get("event_no", 0)
    frame["event"] = np.arange(start + 1, start + len(frame) + 1)
    st.session_state["event_no"] = start + len(frame)
    h = st.session_state["history"]
    st.session_state["history"] = (pd.concat([h, frame], ignore_index=True) if len(h) else frame).tail(5000).reset_index(drop=True)


def membership_figure(var, marker=None):
    cfg = VARIABLES[var]
    xs = np.linspace(0, 1, 400)
    fig = go.Figure()
    for (name, (k, p)), col in zip(MFS[var].items(), [CYAN, PURPLE, ui.AMBER]):
        fig.add_trace(go.Scatter(x=xs * cfg["cap"], y=membership(k, p, xs), name=name, mode="lines",
                                 line=dict(color=col, width=3), fill="tozeroy"))
    if marker is not None:
        fig.add_vline(x=marker, line_dash="dash", line_color="#ffffff")
    return style(fig, cfg["label"], 330, xtitle=cfg["label"], ytitle="Membership degree")


# =============================================================================
# 7. PAGES
# =============================================================================
def page_rules(ev):
    ui.hero("Membership Functions & Rules", "The 3 fuzzy inputs and the 27 rules used by both systems")
    t_mf, t_rules = st.tabs(["Membership functions", "27 fuzzy rules"])
    with t_mf:
        for col, v in zip(st.columns(3), VARIABLES):
            with col:
                show(membership_figure(v))
    with t_rules:
        st.dataframe(pd.DataFrame([dict(ID=r["id"], IF=rule_text(r), Mamdani_THEN=r["mam"] + " risk",
                                        Sugeno_THEN=r["sug"], Purpose=r["why"]) for r in RULES]),
                     width="stretch", hide_index=True, height=640)


def page_compare(ev):
    ui.hero("Mamdani vs Sugeno", "Both systems evaluated on the official test file")
    _, test, _ = get_data()
    R_ = compute_models(ev)
    M = {n: dict(r["m"], AUC=r["auc"], ms=ev["ms_per_1000"][n]) for n, r in R_.items()}

    for col, n in zip(st.columns(2), M):
        with col:
            st.markdown(f"##### {n}")
            a = st.columns(4)
            for i, (lab, k) in enumerate((("Accuracy", "Accuracy"), ("Recall", "Recall"),
                                          ("False-pos. rate", "FPR"), ("AUC", "AUC"))):
                a[i].metric(lab, f"{M[n][k]:.3f}")

    rows = {"Accuracy": "Accuracy", "Precision": "Precision", "Recall": "Recall", "F1-score": "F1",
            "Specificity": "Specificity", "False Positive Rate": "FPR", "False Negative Rate": "FNR",
            "AUC": "AUC", "Time (ms per 1000 rec.)": "ms"}
    st.dataframe(pd.DataFrame({n: {lab: M[n][k] for lab, k in rows.items()} for n in M}).style.format("{:.4f}"),
                 width="stretch")

    keys = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR", "FNR", "AUC"]
    fig = px.bar(pd.DataFrame([dict(Metric=k, Model=n, Value=M[n][k]) for n in M for k in keys]),
                 x="Metric", y="Value", color="Model", barmode="group", text_auto=".3f", color_discrete_map=MODEL_COLORS)
    show(style(fig, "Evaluation metrics (test file)", 420, xtitle="Metric", ytitle="Value (0-1)"))

    c1, c2 = st.columns(2)
    with c1:
        show(ui.confusion_figure(R_["Mamdani"]["c"], "Confusion matrix: Mamdani"))
    with c2:
        show(ui.confusion_figure(R_["Sugeno"]["c"], "Confusion matrix: Sugeno"))

    d1, d2 = st.columns(2)
    fig = go.Figure()
    for n, col in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=R_[n]["fpr"], y=R_[n]["tpr"], mode="lines", line=dict(color=col, width=3),
                                 name=f"{n} (AUC = {R_[n]['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#667", dash="dash"), name="Random guess"))
    with d1:
        show(style(fig, "ROC curves", 400, xtitle="False positive rate", ytitle="True positive rate"))
    fig = px.bar(x=list(M), y=[M[n]["ms"] for n in M], color=list(M), text_auto=".1f", color_discrete_map=MODEL_COLORS)
    fig.update_layout(showlegend=False)
    with d2:
        show(style(fig, "Prediction time", 400, xtitle="Model", ytitle="Milliseconds per 1000 records"))

    st.markdown("### Which model is better?")
    ui.card("<br>".join(verdict_text(M)), "good")

    rows = []
    for n, r in R_.items():
        for f_ in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
            m = (test["family"] == f_).values
            if m.sum():
                rows.append(dict(Family=f_, Model=n, Alert_rate=float((r["cls"][m] != "Normal").mean()), Records=int(m.sum())))
    fig = px.bar(pd.DataFrame(rows), x="Family", y="Alert_rate", color="Model", barmode="group", text_auto=".2f",
                 hover_data=["Records"], color_discrete_map=MODEL_COLORS)
    show(style(fig, "Share of records flagged as alert, per true family", 400, xtitle="True traffic family",
               ytitle="Alert rate (Normal = false alarms, others = detection)"))


def explain_table(m_row, s_row):
    rows = [dict(Rule=r["id"], IF=rule_text(r), Mamdani_strength=round(float(m_row[i]), 3), Mamdani_output=r["mam"],
                 Sugeno_strength=round(float(s_row[i]), 3), Sugeno_output=r["sug"], Meaning=r["why"])
            for i, r in enumerate(RULES) if m_row[i] > 0.001 or s_row[i] > 0.001]
    return pd.DataFrame(rows).sort_values("Mamdani_strength", ascending=False)


def read_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return None, f"Could not read the file: {exc}"
    df.columns = df.columns.str.strip()
    if df.empty or not any(c in df.columns for c in VARIABLES):
        return None, "The file is empty or has none of the columns: " + ", ".join(VARIABLES)
    if "label" in df.columns:
        name = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
        df["actual"] = np.where(name == "normal", "Normal", "Attack")
        df["family"] = name.map(lambda n: "Normal" if n == "normal" else FAMILY.get(n, "Other"))
    return df, None


def page_predict(ev):
    ui.hero("Intrusion Prediction", "Enter the three traffic parameters or upload a CSV file")
    _, test, med = get_data()
    tab_manual, tab_csv = st.tabs(["Manual input", "CSV batch prediction"])

    with tab_manual:
        cols = st.columns(3)
        for col, (v, cfg) in zip(cols, VARIABLES.items()):
            st.session_state.setdefault(f"manual_{v}", float(med[v]))
            col.number_input(cfg["label"], min_value=0.0, max_value=float(cfg["cap"]),
                             step=0.01 if cfg["cap"] <= 1 else 1.0, key=f"manual_{v}", help=cfg["info"])
        df = pd.DataFrame([{v: st.session_state[f"manual_{v}"] for v in VARIABLES}])
        scored, m_fire, s_fire = score_records(df, "manual", ev)
        mc, sc = scored.loc[0, "mamdani_class"], scored.loc[0, "sugeno_class"]

        st.markdown("### Result")
        g1, g2 = st.columns(2)
        with g1:
            show(ui.gauge(scored.loc[0, "mamdani_risk"], "Mamdani risk score", *ev["thr"]["mamdani"]))
            ui.class_label("Mamdani", mc)
        with g2:
            show(ui.gauge(scored.loc[0, "sugeno_risk"], "Sugeno risk score", *ev["thr"]["sugeno"]))
            ui.class_label("Sugeno", sc)
        if mc != sc:
            ui.card(f"<b>Disagreement:</b> Mamdani says <b>{mc}</b>, Sugeno says <b>{sc}</b>.", "bad")
        else:
            ui.card(f"<b>Both systems agree:</b> {mc}.", "good")

        risk, _, agg = mamdani_infer(preprocess(df, med), return_agg=True)
        p1, p2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=UNIVERSE, y=agg[0], fill="tozeroy", line=dict(color=CYAN, width=3)))
        fig.add_vline(x=risk[0], line_dash="dash", line_color="#fff", annotation_text=f"centroid = {risk[0]:.1f}")
        with p1:
            show(style(fig, "Mamdani: aggregated output and centroid", 360, xtitle="Risk score", ytitle="Membership degree"))
        ex = explain_table(m_fire[0], s_fire[0])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ex["Rule"], y=ex["Mamdani_strength"], name="Mamdani", marker_color=CYAN))
        fig.add_trace(go.Bar(x=ex["Rule"], y=ex["Sugeno_strength"], name="Sugeno", marker_color=PURPLE))
        fig.update_layout(barmode="group")
        with p2:
            show(style(fig, "Rules activated by this record", 360, xtitle="Rule", ytitle="Firing strength (0-1)"))
        st.markdown("##### Activated rules")
        st.dataframe(ex, width="stretch", hide_index=True)

        with st.expander("Where does this record sit on the membership functions?"):
            for col, v in zip(st.columns(3), VARIABLES):
                with col:
                    show(membership_figure(v, df[v].iloc[0]))
        if st.button("Submit this record to the Live Monitor"):
            append_history(scored)
            st.success("Record added to the Live Monitor.")

    with tab_csv:
        st.caption("Upload a CSV with the columns: " + ", ".join(VARIABLES) + " (optional: label).")
        st.download_button("Download a sample CSV (50 records)", test.head(50)[list(VARIABLES)].to_csv(index=False),
                           "sample_traffic.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type=["csv", "txt"])
        if up is None:
            return
        df, err = read_csv(up)
        if err:
            st.error(err)
            return
        missing = [c for c in VARIABLES if c not in df.columns]
        if missing:
            st.warning("Missing columns filled with training medians: " + ", ".join(missing))
        scored, _, _ = score_records(df, "csv", ev)
        st.success(f"{len(scored):,} records scored.")
        cnt = pd.DataFrame({"Mamdani": scored["mamdani_class"].value_counts(),
                            "Sugeno": scored["sugeno_class"].value_counts()}).reindex(CLASSES).fillna(0)
        cnt.index.name = "Class"
        cnt = cnt.reset_index().melt("Class", var_name="Model", value_name="Records")
        a, b = st.columns(2)
        with a:
            show(style(px.bar(cnt, x="Class", y="Records", color="Model", barmode="group", text_auto=True,
                              color_discrete_map=MODEL_COLORS), "Predicted classes", 360))
        with b:
            h = scored.melt(value_vars=["mamdani_risk", "sugeno_risk"], var_name="Model", value_name="Risk")
            show(style(px.histogram(h, x="Risk", color="Model", barmode="overlay", nbins=40, opacity=0.65,
                                    color_discrete_map={"mamdani_risk": CYAN, "sugeno_risk": PURPLE}),
                       "Risk score distribution", 360, xtitle="Risk score", ytitle="Records"))
        dis = int((~scored["agree"]).sum())
        st.metric("Records where the two models disagree", f"{dis:,} ({dis / len(scored):.1%})")
        if "actual" in df.columns:
            y = df["actual"].values == "Attack"
            st.markdown("##### Accuracy on the uploaded file (using its label column)")
            st.dataframe(pd.DataFrame({n: metrics_from(confusion(y, scored[k].values != "Normal"))
                                       for n, k in (("Mamdani", "mamdani_class"), ("Sugeno", "sugeno_class"))}
                                      ).style.format("{:.4f}"), width="stretch")
        st.dataframe(scored[["actual", "family", "mamdani_risk", "mamdani_class", "sugeno_risk", "sugeno_class",
                             "agree"]].head(500), width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Download all predictions (CSV)", scored.to_csv(index=False), "predictions.csv", "text/csv")
        if d2.button("Send these records to the Live Monitor"):
            append_history(scored.head(2000))
            st.success("Records added to the Live Monitor.")


def stream_batch(n, ev):
    _, test, _ = get_data()
    append_history(score_records(test.sample(n), "simulation", ev)[0])


def monitor_view(auto, n, ev):
    if auto:
        stream_batch(n, ev)
    h = st.session_state["history"]
    if len(h) == 0:
        st.info("No traffic yet. Press 'Stream next batch', switch on auto-stream, or send records from the Prediction page.")
        return
    st.caption("M / S = Mamdani / Sugeno counts")
    k = st.columns(5)
    k[0].metric("Events", f"{len(h):,}")
    for col, c in zip(k[1:4], CLASSES):
        col.metric(f"{c}  (M / S)", f"{(h['mamdani_class'] == c).sum()} / {(h['sugeno_class'] == c).sum()}")
    k[4].metric("Model disagreements", f"{(~h['agree']).sum()}")

    fig = go.Figure()
    for n_, col in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=h["event"], y=h[n_.lower() + "_risk"], mode="lines+markers", name=f"{n_} risk",
                                 line=dict(color=col)))
    show(style(fig, "Intrusion risk score of every event", 420, xtitle="Event number", ytitle="Risk score (0-100)"))

    a, b = st.columns(2)
    cnt = pd.DataFrame({"Mamdani": h["mamdani_class"].value_counts(), "Sugeno": h["sugeno_class"].value_counts()}) \
        .reindex(CLASSES).fillna(0)
    cnt.index.name = "Class"
    with a:
        show(style(px.bar(cnt.reset_index().melt("Class", var_name="Model", value_name="Events"), x="Class", y="Events",
                          color="Model", barmode="group", text_auto=True, color_discrete_map=MODEL_COLORS),
                   "Traffic classes: Mamdani vs Sugeno", 360))
    ag = h["agree"].map({True: "Models agree", False: "Models disagree"}).value_counts().reset_index()
    ag.columns = ["Result", "Events"]
    with b:
        show(style(px.pie(ag, names="Result", values="Events", hole=0.55, color="Result",
                          color_discrete_map={"Models agree": GREEN, "Models disagree": RED}),
                   "Agreement between the models", 360))

    tr = h.groupby("batch").agg(Mamdani=("mamdani_class", lambda s: (s != "Normal").sum()),
                                Sugeno=("sugeno_class", lambda s: (s != "Normal").sum())).reset_index()
    fig = px.line(tr.melt("batch", var_name="Model", value_name="Alerts"), x="batch", y="Alerts", color="Model",
                  markers=True, color_discrete_map=MODEL_COLORS)
    show(style(fig, "Alerts per batch", 360, xtitle="Batch number", ytitle="Alerts (Suspicious + Malicious)"))

    st.markdown("##### Prediction history (latest 100)")
    cols = ["event", "time", "source", "actual", "family", "mamdani_risk", "mamdani_class", "sugeno_risk",
            "sugeno_class", "agree"]
    st.dataframe(h[cols].tail(100).iloc[::-1], width="stretch", hide_index=True)


def page_monitor(ev):
    ui.hero("Live Monitor", "Test traffic, uploaded files and manual records scored by both fuzzy systems")
    ui.card("<b>Simulation:</b> not connected to a real network. Random records from the test file are replayed.", "sim")
    c = st.columns(4)
    n = c[0].slider("Records per batch", 1, 50, 10)
    auto = c[1].toggle("Auto-stream every 2 s", value=False)
    if c[2].button("Stream next batch", width="stretch"):
        stream_batch(n, ev)
    if c[3].button("Clear history", width="stretch"):
        st.session_state.update(history=pd.DataFrame(), event_no=0, batch_no=0)
    (st.fragment(run_every=2)(monitor_view) if auto else st.fragment(monitor_view))(auto, n, ev)


def page_analytics(ev):
    ui.hero("Analytics", "Score distributions, rule activation and trends")
    _, test, _ = get_data()
    t1, t2, t3 = st.tabs(["Risk score distribution", "Rule activation", "Prediction trends"])
    with t1:
        long = pd.concat([pd.DataFrame({"Risk": ev["test"][k + "_risk"].values, "Model": n, "Actual": test["actual"].values})
                          for n, k in (("Mamdani", "mamdani"), ("Sugeno", "sugeno"))])
        fig = px.histogram(long, x="Risk", color="Actual", facet_col="Model", barmode="overlay", nbins=50, opacity=0.7,
                           color_discrete_map={"Normal": GREEN, "Attack": RED})
        show(style(fig, "Risk score distribution by true class (test file)", 450, xtitle="Risk score", ytitle="Records"))
    with t2:
        heat = pd.DataFrame(ev["m_fire"], columns=[r["id"] for r in RULES]).groupby(test["family"].values).mean()
        fig = px.imshow(heat.round(2), text_auto=True, aspect="auto", color_continuous_scale=[[0, "#0b1230"], [1, CYAN]])
        show(style(fig, "Average Mamdani rule activation per true traffic family (test file)", 480,
                   xtitle="Rule", ytitle="True family"))
    with t3:
        h = st.session_state["history"]
        if len(h) == 0:
            st.info("No monitor data yet. Stream records in the Live Monitor or submit predictions first.")
        else:
            h = h.copy()
            h["Mamdani alerts (rolling 20)"] = (h["mamdani_class"] != "Normal").rolling(20, min_periods=1).sum()
            h["Sugeno alerts (rolling 20)"] = (h["sugeno_class"] != "Normal").rolling(20, min_periods=1).sum()
            fig = px.line(h, x="event", y=["Mamdani alerts (rolling 20)", "Sugeno alerts (rolling 20)"],
                          color_discrete_sequence=[CYAN, PURPLE])
            show(style(fig, "Alerts over time (rolling window of 20 events)", 420, xtitle="Event number",
                       ytitle="Alerts in window"))


# =============================================================================
# 8. MAIN
# =============================================================================
PAGES = {"Rules & Membership": page_rules, "Mamdani vs Sugeno": page_compare, "Prediction": page_predict,
         "Live Monitor": page_monitor, "Analytics": page_analytics}


def main():
    st.set_page_config(page_title="Fuzzy IDS", page_icon="🛡️", layout="wide")
    ui.apply_style()
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        st.error("Dataset not found. Place KDDTrain_.txt and KDDTest_.txt inside the 'data' folder next to app.py.")
        st.stop()
    st.session_state.setdefault("history", pd.DataFrame())
    ev = evaluate()
    with st.sidebar:
        st.markdown("## 🛡️ Fuzzy IDS")
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    PAGES[page](ev)


if __name__ == "__main__":
    main()
