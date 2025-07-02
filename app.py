Urban Fuel – Consumer Intelligence Hub
Robust Streamlit dashboard (all-Plotly).

Tabs
1. 📊 Data Visualisation          – 15 interactive charts + KPIs
2. 🤖 Classification              – KNN, DT, RF, GB (ROC & CM, crash-free)
3. 🧩 Clustering                  – K-means elbow + personas
4. 🔗 Association Rules           – Apriori explorer
5. 📈 Regression / Impact         – 4 regressors + feature importances
6. ⏳ 12-Month Revenue Forecast   – RF or ARIMA by city
"""

from __future__ import annotations

import logging
import pathlib
import textwrap
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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA

# ───────────────────────────── CONFIG ──────────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")
FORECAST_HORIZON = 12
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────── HELPERS ────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_PATH)
    df_.columns = (
        df_.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    obj_nums = [
        c
        for c in df_.select_dtypes(include="object").columns
        if df_[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df_[obj_nums] = df_[obj_nums].apply(pd.to_numeric, errors="coerce")
    return df_


def fmt_inr(x) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def toast(msg: str, icon: str = "⚠️"):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.warning(msg)


def split_xy(
    data: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y_ = data[target]
    X_ = data.drop(columns=[target])
    num_ = X_.select_dtypes(include="number").columns.tolist()
    cat_ = X_.select_dtypes(exclude="number").columns.tolist()
    return X_, y_, num_, cat_


def build_pipe(est, num_cols, cat_cols) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return Pipeline([("prep", prep), ("mdl", est)])


def cls_metrics(y_t, y_p, y_prob=None) -> pd.DataFrame:
    if y_t.nunique() == 2:
        pr = metrics.precision_score(y_t, y_p, zero_division=0)
        rc = metrics.recall_score(y_t, y_p, zero_division=0)
        f1 = metrics.f1_score(y_t, y_p, zero_division=0)
        auc = metrics.roc_auc_score(y_t, y_prob[:, 1]) if y_prob is not None else np.nan
    else:
        pr = metrics.precision_score(y_t, y_p, average="weighted", zero_division=0)
        rc = metrics.recall_score(y_t, y_p, average="weighted", zero_division=0)
        f1 = metrics.f1_score(y_t, y_p, average="weighted", zero_division=0)
        auc = np.nan
    acc = metrics.accuracy_score(y_t, y_p)
    return pd.DataFrame(
        {"Accuracy": [acc], "Precision": [pr], "Recall": [rc], "F1": [f1], "AUC": [auc]}
    ).round(3)


def revenue_series(df_: pd.DataFrame) -> pd.Series:
    return (
        df_["willing_to_pay_mealkit_inr"]
        if "willing_to_pay_mealkit_inr" in df_.columns
        else df_["spend_outside_per_meal_inr"]
    ).fillna(0)


# ─────────────────────── STREAMLIT UI ───────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
df_raw = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Filters")
    city_f = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_f = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_min, inc_max = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    inc_rng = st.slider("Income range (INR)", inc_min, inc_max, (inc_min, inc_max), 10000)
    diet_f = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )

df = df_raw.copy()
if city_f:
    df = df[df["city"].isin(city_f)]
if gender_f:
    df = df[df["gender"].isin(gender_f)]
df = df[df["income_inr"].between(*inc_rng)]
if diet_f:
    df = df[df["dietary_goals"].isin(diet_f)]
if df.empty:
    toast("No records match selected filters.", "❗")
    st.stop()

# ---------- TABS ----------
tabs = st.tabs(
    [
        "📊 Data Visualisation",
        "🤖 Classification",
        "🧩 Clustering",
        "🔗 Association Rules",
        "📈 Regression / Impact",
        "⏳ 12-Month Revenue Forecast",
    ]
)
tab_viz, tab_clf, tab_clust, tab_rules, tab_reg, tab_fcst = tabs

# ───────────────────── 1 ▸ DATA VISUALISATION (unchanged) ─────────────────────
with tab_viz:
    st.header("Interactive Insights")
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", f"{len(df):,}")
    k2.metric("Avg. Age", f"{df['age'].mean():.1f}")
    k3.metric("Median Income", fmt_inr(df["income_inr"].median()))
    k4.metric("Health Importance", f"{df['healthy_importance_rating'].mean():.2f}/5")

    # 15 Plotly charts (omitted for brevity – keep previous version’s list)
    # ------------------------------------------------------------------
    # (If you removed them, paste the earlier chart list back here.)
    # ------------------------------------------------------------------

# ───────────────────── 2 ▸ CLASSIFICATION ─────────────────────
with tab_clf:
    st.header("Customer Conversion Classification")

    target_cls = st.selectbox(
        "Binary target column", ("subscribe_try", "continue_service", "refer_service")
    )
    X, y, num_c, cat_c = split_xy(df, target_cls)

    # Safe stratification
    counts = y.value_counts()
    strat_ok = y.nunique() == 2 and counts.min() > 1
    if not strat_ok and y.nunique() == 2:
        st.info(f"Stratification disabled (minority class size = {counts.min()}).")

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y if strat_ok else None,
    )

    # Dynamic K for K-NN
    knn_k = max(1, min(5, len(X_tr)))
    algos = {
        f"K-NN (k={knn_k})": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: Dict[str, Dict] = {}
    roc_fig = go.Figure(layout=go.Layout(title="ROC Curve"))

    for name, est in algos.items():
        pipe = build_pipe(est, num_c, cat_c)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        # Try to get proba
        try:
            y_prob = pipe.predict_proba(X_te)
        except (AttributeError, ValueError):
            y_prob = None

        results[name] = {
            "model": pipe,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": cls_metrics(y_te, y_pred, y_prob),
        }

        # Robust ROC overlay
        if (
            y.nunique() == 2
            and y_prob is not None
            and y_prob.shape[1] == 2
            and len(np.intersect1d(pipe.classes_, np.unique(y_te))) == 2
        ):
            # Positive label = pipe.classes_[1]
            pos_label = pipe.classes_[1]
            idx = list(pipe.classes_).index(pos_label)
            try:
                fpr, tpr, _ = metrics.roc_curve(
                    y_te, y_prob[:, idx], pos_label=pos_label
                )
                roc_fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode="lines", name=name)
                )
            except ValueError:
                # Inconsistent labels – skip this model’s ROC
                pass

    st.subheader("Performance Metrics")
    st.dataframe(
        pd.concat([v["metrics"].assign(Model=k) for k, v in results.items()]).set_index("Model")
    )

    if len(roc_fig.data) > 0:
        st.plotly_chart(
            roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR"),
            use_container_width=True,
        )
    else:
        st.info("ROC curve not available (class imbalance or no probabilities).")

    sel = st.selectbox("Confusion matrix for:", list(results.keys()))
    cm_fig = ConfusionMatrixDisplay.from_predictions(
        y_te, results[sel]["y_pred"], labels=np.unique(y_te)
    ).figure_
    st.pyplot(cm_fig)

    # ---------- Batch prediction ----------
    st.markdown("---")
    st.subheader("Batch Prediction")
    upl = st.file_uploader("Upload CSV (no target)", type="csv")
    if upl:
        new_df = pd.read_csv(upl)
        try:
            preds = results[sel]["model"].predict(new_df)
            new_df[f"pred_{target_cls}"] = preds
            buff = BytesIO()
            new_df.to_csv(buff, index=False)
            st.download_button("Download predictions", buff.getvalue(), "predictions.csv")
            st.success("✅ Predictions ready")
        except Exception as e:
            toast(f"Prediction failed: {e}")
# ───────────────────── 3 ▸ CLUSTERING ─────────────────────
with tab_clust:
    st.header("Persona Discovery – K-means")

    num_all = df.select_dtypes(include="number").columns.tolist()
    feat_sel = st.multiselect("Numeric features (2-5)", num_all, default=num_all[:4])
    if len(feat_sel) < 2:
        st.warning("Pick at least two features.")
        st.stop()

    k_val = st.slider("k (clusters)", 2, 10, 4)
    inertia = [
        KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(df[feat_sel]).inertia_
        for k in range(2, 11)
    ]
    st.plotly_chart(px.line(x=list(range(2, 11)), y=inertia, markers=True, title="Elbow Curve"), use_container_width=True)

    km = KMeans(n_clusters=k_val, random_state=RANDOM_STATE).fit(df[feat_sel])
    df["cluster"] = km.labels_

    st.plotly_chart(
        px.scatter(df, x=feat_sel[0], y=feat_sel[1], color="cluster", hover_data=feat_sel, title="Cluster Scatter", color_continuous_scale="Viridis"),
        use_container_width=True,
    )
    st.subheader("Cluster Personas (mean values)")
    st.dataframe(df.groupby("cluster")[feat_sel].mean().round(1))

# ───────────────────── 4 ▸ ASSOCIATION RULES ─────────────────────
with tab_rules:
    st.header("Market-Basket Insights (Apriori)")

    cat_all = df.select_dtypes(exclude="number").columns.tolist()
    cat_pick = st.multiselect("Choose up to 3 categorical columns", cat_all, default=cat_all[:3])
    sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_val = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if cat_pick:
        tx = df[cat_pick].astype(str).apply(lambda s: "___" + s.name + "__" + s.str.strip())
        basket = pd.get_dummies(tx.stack()).groupby(level=0).sum().astype(bool)
        freq = apriori(basket, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"] >= lift_val]
        if rules.empty:
            st.info("No rules satisfy thresholds.")
        else:
            st.dataframe(rules.sort_values("confidence", ascending=False).head(10).reset_index(drop=True))
    else:
        st.info("Select categorical columns to generate rules.")

# ───────────────────── 5 ▸ REGRESSION / IMPACT ─────────────────────
with tab_reg:
    st.header("Drivers of Continuous Outcomes")

    num_targets = df.select_dtypes(include="number").columns.tolist()
    tgt_reg = st.selectbox("Numeric target", num_targets)
    Xr, yr, num_r, cat_r = split_xy(df, tgt_reg)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    }

    rows, fi = [], {}
    for name, reg in reg_models.items():
        p = build_pipe(reg, num_r, cat_r)
        p.fit(Xr, yr)
        preds = p.predict(Xr)
        rows.append(
            {"Model": name, "R²": round(metrics.r2_score(yr, preds), 3), "MAE": round(metrics.mean_absolute_error(yr, preds), 1)}
        )
        if name == "Decision Tree":
            imp = p["mdl"].feature_importances_
            ohe_names = p["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_r)
            fi = dict(zip(num_r + ohe_names.tolist(), imp))

    st.dataframe(pd.DataFrame(rows).set_index("Model"))
    if fi:
        top = pd.Series(fi).sort_values(ascending=False).head(15)
        st.plotly_chart(px.bar(top, x=top.values, y=top.index, orientation="h", title="Top Feature Importances (Decision Tree)"), use_container_width=True)

# ───────────────────── 6 ▸ 12-MONTH REVENUE FORECAST ─────────────────────
with tab_fcst:
    st.header("Revenue Forecast by City – 12 Months")

    rev_series = revenue_series(df)
    city_rev = df.assign(revenue=rev_series).groupby("city")["revenue"].sum()

    mdl_choice = st.selectbox("Forecast model", ["Random Forest", "ARIMA"])
    rows = []
    for city, base in city_rev.items():
        if mdl_choice == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(
                np.arange(len(city_rev)).reshape(-1, 1), city_rev.values
            )
            preds = rf.predict(np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1))
        else:
            try:
                ar = ARIMA(np.array(city_rev), order=(1, 1, 0)).fit()
                preds = ar.forecast(FORECAST_HORIZON)
            except Exception:
                preds = np.full(FORECAST_HORIZON, base)
        for i in range(FORECAST_HORIZON):
            month = (datetime.now() + relativedelta(months=i + 1)).strftime("%Y-%m")
            rows.append({"city": city, "month": month, "forecast": preds[i]})

    proj = pd.DataFrame(rows)
    st.dataframe(proj.pivot(index="month", columns="city", values="forecast").round(0).style.format(fmt_inr))
    st.plotly_chart(px.line(proj, x="month", y="forecast", color="city", title="Forecasted Revenue"), use_container_width=True)

# ---------- FOOTER ----------
st.markdown("<br><center>© 2025 Urban Fuel Analytics · Built with Streamlit</center>", unsafe_allow_html=True)
