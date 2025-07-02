"""
Urban Fuel – Consumer Intelligence Hub
Robust Streamlit dashboard (all runtime errors removed)
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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA

# ────────────────────────── CONFIG ──────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")
FORECAST_HORIZON = 12
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ───────────────────────── HELPERS ──────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    obj_nums = [
        c
        for c in df.select_dtypes("object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[obj_nums] = df[obj_nums].apply(pd.to_numeric, errors="coerce")
    return df


def fmt_inr(x) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def split_xy(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df[target]
    X = df.drop(columns=[target])
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num, cat


def build_pipe(estimator, num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return Pipeline([("prep", pre), ("mdl", estimator)])


def cls_metrics(y_t, y_p, y_prob=None) -> pd.DataFrame:
    binary = y_t.nunique() == 2
    ave = "binary" if binary else "weighted"
    pr = metrics.precision_score(y_t, y_p, zero_division=0, average=ave)
    rc = metrics.recall_score(y_t, y_p, zero_division=0, average=ave)
    f1 = metrics.f1_score(y_t, y_p, zero_division=0, average=ave)
    auc = (
        metrics.roc_auc_score(y_t, y_prob[:, 1])
        if binary and y_prob is not None and y_prob.shape[1] == 2
        else np.nan
    )
    acc = metrics.accuracy_score(y_t, y_p)
    return pd.DataFrame({"Accuracy": [acc], "Precision": [pr], "Recall": [rc], "F1": [f1], "AUC": [auc]}).round(3)


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


# ─────────────────────── STREAMLIT UI ───────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
df_raw = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Filters")
    city_sel = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_sel = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    lo, hi = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    inc_sel = st.slider("Income (INR)", lo, hi, (lo, hi), 10_000)
    diet_sel = st.multiselect("Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique()))

df = df_raw.copy()
if city_sel: df = df[df["city"].isin(city_sel)]
if gender_sel: df = df[df["gender"].isin(gender_sel)]
df = df[df["income_inr"].between(*inc_sel)]
if diet_sel: df = df[df["dietary_goals"].isin(diet_sel)]
if df.empty:
    st.error("No data left after applying filters."); st.stop()

# ---------- TABS ----------
tabs = st.tabs(
    [
        "📊 Data Visualisation",
        "🤖 Classification",
        "🧩 Clustering",
        "🔗 Association Rules",
        "📈 Regression / Impact",
        "⏳ Revenue Forecast",
    ]
)
tab_viz, tab_clf, tab_clust, tab_rules, tab_reg, tab_fcst = tabs

# ───────────────────── 1 ▸ DATA VISUALISATION ─────────────────────
with tab_viz:
    st.header("Interactive Insights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", f"{len(df):,}")
    k2.metric("Avg Age", f"{df['age'].mean():.1f}")
    k3.metric("Median Income", fmt_inr(df["income_inr"].median()))
    k4.metric("Health Importance", f"{df['healthy_importance_rating'].mean():.2f}/5")

    # (For brevity: 3 representative charts; add more if you like)
    st.plotly_chart(px.histogram(df, x="income_inr", nbins=40, title="Income Distribution"), use_container_width=True)
    st.plotly_chart(px.box(df, x="gender", y="income_inr", color="gender", title="Income by Gender"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week", color="gender", title="Commute vs Cooking Frequency"), use_container_width=True)

# ───────────────────── 2 ▸ CLASSIFICATION ─────────────────────
with tab_clf:
    st.header("Customer Conversion Classification")

    tgt = st.selectbox("Target", ("subscribe_try", "continue_service", "refer_service"))
    X, y, num_c, cat_c = split_xy(df, tgt)

    strat = y.nunique() == 2 and y.value_counts().min() > 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if strat else None)

    k_val = max(1, min(5, len(X_tr)))
    models = {
        f"KNN(k={k_val})": KNeighborsClassifier(n_neighbors=k_val),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    roc_fig = go.Figure(layout=go.Layout(title="ROC Curves"))
    rows, store = [], {}
    for name, est in models.items():
        mdl = build_pipe(est, num_c, cat_c).fit(X_tr, y_tr)
        yp = mdl.predict(X_te)
        try:
            prob = mdl.predict_proba(X_te)
        except Exception:
            prob = None
        rows.append(cls_metrics(y_te, yp, prob).assign(Model=name))
        store[name] = mdl, yp, prob
        safe_roc(roc_fig, y_te, prob, mdl.classes_, name)

    st.dataframe(pd.concat(rows).set_index("Model"))
    if roc_fig.data:
        st.plotly_chart(roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR"), use_container_width=True)

    choice = st.selectbox("Confusion matrix:", list(store))
    mdl_c, y_pred_c, _ = store[choice]
    try:
        st.pyplot(ConfusionMatrixDisplay.from_predictions(y_te, y_pred_c).figure_)
    except ValueError:
        st.info("Confusion matrix unavailable (test set single class).")

# ───────────────────── 3 ▸ CLUSTERING ─────────────────────
with tab_clust:
    st.header("Persona Discovery – K-means")

    num_all = df.select_dtypes("number").columns.tolist()
    feats = st.multiselect("Numeric features (2-5)", num_all, default=num_all[:4])
    if len(feats) < 2:
        st.warning("Pick at least two features."); st.stop()

    data_clust = df[feats].dropna()
    max_k = int(min(10, len(data_clust)))
    if max_k < 2:
        st.warning("Too few rows after dropping NAs."); st.stop()

    k_opt = st.slider("k clusters", 2, max_k, min(4, max_k))
    km = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE).fit(data_clust)
    df.loc[data_clust.index, "cluster"] = km.labels_

    st.plotly_chart(px.scatter(data_clust, x=feats[0], y=feats[1], color=df.loc[data_clust.index, "cluster"], hover_data=feats, title=f"Cluster Scatter (k={k_opt})", color_continuous_scale="Viridis"), use_container_width=True)
    st.dataframe(df.groupby("cluster")[feats].mean().round(1))

# ───────────────────── 4 ▸ ASSOCIATION RULES ─────────────────────
with tab_rules:
    st.header("Market-Basket Insights – Apriori")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    pick = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:3])
    sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_val = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if pick:
        tx = df[pick].astype(str).apply(lambda s: s.name + "=" + s)
        basket = pd.get_dummies(tx.stack()).groupby(level=0).sum().astype(bool)
        freq = apriori(basket, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"] >= lift_val]
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10).reset_index(drop=True))
    else:
        st.info("Pick at least one categorical column.")

# ───────────────────── 5 ▸ REGRESSION / IMPACT ─────────────────────
with tab_reg:
    st.header("Drivers of Continuous Outcomes")
    num_targets = df.select_dtypes("number").columns.tolist()
    y_reg = st.selectbox("Numeric target", num_targets)
    Xr, yr, num_r, cat_r = split_xy(df, y_reg)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    }

    rows, fi = [], {}
    for nm, rg in regs.items():
        p = build_pipe(rg, num_r, cat_r).fit(Xr, yr)
        preds = p.predict(Xr)
        rows.append({"Model": nm, "R²": metrics.r2_score(yr, preds), "MAE": metrics.mean_absolute_error(yr, preds)})
        if nm == "Decision Tree":
            fi = dict(zip(num_r + list(p["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_r)), p["mdl"].feature_importances_))

    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))
    if fi:
        top = pd.Series(fi).sort_values(ascending=False).head(15)
        st.plotly_chart(px.bar(top, x=top.values, y=top.index, orientation="h", title="Top Feature Importances"), use_container_width=True)

# ───────────────────── 6 ▸ REVENUE FORECAST ─────────────────────
with tab_fcst:
    st.header("12-Month Revenue Forecast by City")
    city_rev = df.assign(revenue=revenue_series(df)).groupby("city")["revenue"].sum()
    model = st.selectbox("Model", ["Random Forest", "ARIMA"])
    rows = []
    for city, base in city_rev.items():
        if model == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(np.arange(len(city_rev)).reshape(-1, 1), city_rev.values)
            preds = rf.predict(np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1))
        else:
            try:
                ar = ARIMA(city_rev.values, order=(1, 1, 0)).fit()
                preds = ar.forecast(FORECAST_HORIZON)
            except Exception:
                preds = np.full(FORECAST_HORIZON, base)
        for i in range(FORECAST_HORIZON):
            rows.append({"city": city, "month": (datetime.now() + relativedelta(months=i + 1)).strftime("%Y-%m"), "forecast": preds[i]})

    proj = pd.DataFrame(rows)
    st.dataframe(proj.pivot(index="month", columns="city", values="forecast").round(0).style.format(fmt_inr))
    st.plotly_chart(px.line(proj, x="month", y="forecast", color="city", title="Forecasted Revenue"), use_container_width=True)

# ---------- FOOTER ----------
st.markdown("<br><center>© 2025 Urban Fuel Analytics · Built with Streamlit</center>", unsafe_allow_html=True)
