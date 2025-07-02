"""
Urban Fuel – Consumer Intelligence Hub
Rock-solid Streamlit dashboard.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA

# ───────────────────────── CONFIG ─────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")
FORECAST_HORIZON = 12
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ────────────────────── DATA LOADER ──────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    obj_to_num = [
        c for c in df.select_dtypes("object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[obj_to_num] = df[obj_to_num].apply(pd.to_numeric, errors="coerce")
    return df


# ──────────────────── GENERIC HELPERS ────────────────────
def fmt_inr(x) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df[target]
    X = df.drop(columns=[target])
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num, cat


def build_pipe(est, num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return Pipeline([("prep", pre), ("mdl", est)])


# ────────────── SAFE PRECISION / RECALL / F1 ─────────────
def safe_prf(y_true, y_pred) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1.
    • If binary and both classes present → binary metrics.
    • Else → weighted metrics (never errors).
    """
    binary = y_true.nunique() == 2
    if binary:
        pos_label = sorted(y_true.unique())[-1]
        try:
            pr = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            rc = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            return pr, rc, f1
        except ValueError:
            pass  # fall through to weighted
    pr = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return pr, rc, f1


def cls_metrics(y_t, y_p, y_prob=None) -> pd.DataFrame:
    pr, rc, f1 = safe_prf(y_t, y_p)
    auc = (
        metrics.roc_auc_score(y_t, y_prob[:, 1])
        if y_t.nunique() == 2 and y_prob is not None and y_prob.shape[1] == 2
        else np.nan
    )
    acc = metrics.accuracy_score(y_t, y_p)
    return pd.DataFrame(
        {"Accuracy": [acc], "Precision": [pr], "Recall": [rc], "F1": [f1], "AUC": [auc]}
    ).round(3)


def safe_roc(fig, y_true, y_prob, classes, label):
    if y_true.nunique() != 2 or y_prob is None or y_prob.shape[1] != 2:
        return
    pos = classes[1]
    if pos not in y_true.values:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1], pos_label=pos)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
    except ValueError:
        pass


def revenue_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["willing_to_pay_mealkit_inr"]
        if "willing_to_pay_mealkit_inr" in df.columns
        else df["spend_outside_per_meal_inr"]
    ).fillna(0)


# ───────────────────── STREAMLIT APP ─────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
df_raw = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Filters")
    city_f = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_f = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    lo, hi = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    inc_f = st.slider("Income (INR)", lo, hi, (lo, hi), 10_000)
    diet_f = st.multiselect("Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique()))

df = df_raw.copy()
if city_f:   df = df[df["city"].isin(city_f)]
if gender_f: df = df[df["gender"].isin(gender_f)]
df = df[df["income_inr"].between(*inc_f)]
if diet_f:   df = df[df["dietary_goals"].isin(diet_f)]
if df.empty:
    st.error("No data after filters."); st.stop()

# ---------- TABS ----------
tabs = st.tabs(
    ["📊 Viz", "🤖 Class", "🧩 Cluster", "🔗 Rules", "📈 Regr", "⏳ Forecast"]
)
tab_viz, tab_clf, tab_clust, tab_rules, tab_reg, tab_fcst = tabs

# ────────────────── 1 ▸ DATA VISUALISATION ──────────────────
with tab_viz:
    st.header("Income Distribution")
    st.plotly_chart(px.histogram(df, x="income_inr", nbins=40), use_container_width=True)

# ────────────────── 2 ▸ CLASSIFICATION ──────────────────
with tab_clf:
    st.header("Classification")
    tgt = st.selectbox("Target", ("subscribe_try", "continue_service", "refer_service"))
    X, y, num_c, cat_c = split_xy(df, tgt)

    strat_ok = y.nunique() == 2 and y.value_counts().min() > 1
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if strat_ok else None
    )

    k_val = max(1, min(5, len(Xtr)))
    algos = {
        f"KNN(k={k_val})": KNeighborsClassifier(n_neighbors=k_val),
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(random_state=RANDOM_STATE),
        "GB": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    roc_fig, rows, store = go.Figure(), [], {}
    for name, est in algos.items():
        m = build_pipe(est, num_c, cat_c).fit(Xtr, ytr)
        yp = m.predict(Xte)
        try:
            yprob = m.predict_proba(Xte)
        except Exception:
            yprob = None
        rows.append(cls_metrics(yte, yp, yprob).assign(Model=name))
        store[name] = (m, yp)
        safe_roc(roc_fig, yte, yprob, m.classes_, name)

    st.dataframe(pd.concat(rows).set_index("Model"))
    if roc_fig.data:
        st.plotly_chart(roc_fig.update_layout(title="ROC Curves"), use_container_width=True)
    else:
        st.info("ROC curves unavailable.")

    choice = st.selectbox("Confusion matrix for:", list(store))
    try:
        st.pyplot(ConfusionMatrixDisplay.from_predictions(yte, store[choice][1]).figure_)
    except ValueError:
        st.info("Confusion matrix unavailable (single-class test split).")

# ────────────────── 3 ▸ CLUSTERING ──────────────────
with tab_clust:
    st.header("K-means Personas")
    num_cols = df.select_dtypes("number").columns.tolist()
    sel = st.multiselect("Numeric features (2-5)", num_cols, default=num_cols[:4])
    if len(sel) < 2:
        st.warning("Pick ≥2 numeric features."); st.stop()
    data = df[sel].dropna()
    max_k = int(min(10, len(data)))
    if max_k < 2:
        st.warning("Too few rows for clustering."); st.stop()
    k = st.slider("k", 2, max_k, min(4, max_k))
    if k > len(data):
        k = len(data)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(data)
    df.loc[data.index, "cluster"] = km.labels_
    st.plotly_chart(px.scatter(data, x=sel[0], y=sel[1], color=df.loc[data.index, "cluster"], hover_data=sel, title=f"Scatter (k={k})"), use_container_width=True)
    st.dataframe(df.groupby("cluster")[sel].mean().round(1))

# ────────────────── 4 ▸ ASSOCIATION RULES ──────────────────
with tab_rules:
    st.header("Apriori Rules")
    cats = df.select_dtypes(exclude="number").columns.tolist()
    if not cats:
        st.info("No categorical columns."); st.stop()
    chosen = st.multiselect("Use columns", cats, default=cats[:3])
    if not chosen:
        st.info("Select ≥1 categorical column."); st.stop()
    sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_v = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    tx = df[chosen].astype(str).apply(lambda s: s.name + "=" + s)
    basket = pd.get_dummies(tx.stack()).groupby(level=0).sum().astype(bool)
    freq = apriori(basket, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] >= lift_v]
    st.dataframe(rules.sort_values("confidence", ascending=False).head(10).reset_index(drop=True) if not rules.empty else pd.DataFrame({"No rules": []}))

# ────────────────── 5 ▸ REGRESSION / IMPACT ──────────────────
with tab_reg:
    st.header("Regression / Feature Impact")
    nums = df.select_dtypes("number").columns.tolist()
    if not nums: st.info("No numeric targets."); st.stop()
    target_num = st.selectbox("Target", nums)
    Xr, yr, num_r, cat_r = split_xy(df, target_num)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "DT": DecisionTreeRegressor(random_state=RANDOM_STATE),
    }

    recs, fi = [], {}
    for nm, rg in regs.items():
        p = build_pipe(rg, num_r, cat_r).fit(Xr, yr)
        recs.append(
            {"Model": nm,
             "R²": metrics.r2_score(yr, p.predict(Xr)),
             "MAE": metrics.mean_absolute_error(yr, p.predict(Xr))}
        )
        if nm == "DT":
            feat_names = num_r + list(p["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_r)) if cat_r else num_r
            fi = dict(zip(feat_names, p["mdl"].feature_importances_))

    st.dataframe(pd.DataFrame(recs).set_index("Model").round(3))
    if fi:
        top = pd.Series(fi).sort_values(ascending=False).head(15)
        st.plotly_chart(px.bar(top, x=top.values, y=top.index, orientation="h", title="Top Feature Importances"), use_container_width=True)

# ────────────────── 6 ▸ REVENUE FORECAST ──────────────────
with tab_fcst:
    st.header("12-Month Revenue Forecast")
    city_rev = df.assign(revenue=revenue_series(df)).groupby("city")["revenue"].sum()
    if city_rev.empty:
        st.info("No revenue data."); st.stop()
    mdl = st.selectbox("Model", ["Random Forest", "ARIMA"])
    rows = []
    for city, base in city_rev.items():
        if mdl == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(np.arange(len(city_rev)).reshape(-1, 1), city_rev.values)
            preds = rf.predict(np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1))
        else:
            try:
                ar = ARIMA(city_rev.values, order=(1, 1, 0)).fit()
                preds = ar.forecast(FORECAST_HORIZON)
            except Exception:
                preds = np.full(FORECAST_HORIZON, base)
        for i in range(FORECAST_HORIZON):
            rows.append({"city": city, "month": (datetime.now() + relativedelta(months=i+1)).strftime("%Y-%m"), "forecast": preds[i]})

    proj = pd.DataFrame(rows)
    st.dataframe(proj.pivot(index="month", columns="city", values="forecast").round(0).style.format(fmt_inr))
    st.plotly_chart(px.line(proj, x="month", y="forecast", color="city"), use_container_width=True)

# ---------- FOOTER ----------
st.markdown("<br><center>© 2025 Urban Fuel Analytics · Built with Streamlit</center>", unsafe_allow_html=True)
