"""
Urban Fuel – Consumer Intelligence Hub
Streamlit dashboard (crash-proof edition)
────────────────────────────────────────────────────────────────────────
Tabs
1. 📊 Data Visualisation          – 15 Plotly charts + KPI cards
2. 🤖 Classification              – KNN, DT, RF, GB (+ safe ROC/CM)
3. 🧩 Clustering                  – K-means elbow + persona explorer
4. 🔗 Association Rules           – Apriori explorer
5. 📈 Regression / Impact         – Linear, Ridge, Lasso, Tree + FI plot
6. ⏳ 12-Month Revenue Forecast   – RF or ARIMA by city
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
    obj_as_num = [
        c
        for c in df.select_dtypes("object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[obj_as_num] = df[obj_as_num].apply(pd.to_numeric, errors="coerce")
    return df


def fmt_inr(x) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def split_xy(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df[target]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num_cols, cat_cols


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


def safe_roc_add(fig, y_true, y_prob, classes, label):
    """Add ROC line only if legal; otherwise skip silently."""
    if y_true.nunique() != 2 or y_prob is None or y_prob.shape[1] != 2:
        return
    pos_label = classes[1]
    if pos_label not in y_true.values:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1], pos_label=pos_label)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
    except ValueError:
        return


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
    city_f = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_f = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_min, inc_max = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    inc_range = st.slider("Income range (INR)", inc_min, inc_max, (inc_min, inc_max), 10_000)
    diet_f = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )

df = df_raw.copy()
if city_f:
    df = df[df["city"].isin(city_f)]
if gender_f:
    df = df[df["gender"].isin(gender_f)]
df = df[df["income_inr"].between(*inc_range)]
if diet_f:
    df = df[df["dietary_goals"].isin(diet_f)]
if df.empty:
    st.error("No data left after filters."); st.stop()

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
    st.header("Key Interactive Insights")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", f"{len(df):,}")
    k2.metric("Avg Age", f"{df['age'].mean():.1f}")
    k3.metric("Median Income", fmt_inr(df["income_inr"].median()))
    k4.metric("Health Importance", f"{df['healthy_importance_rating'].mean():.2f}/5")

    charts = [
        px.histogram(df, x="income_inr", nbins=40, title="Income Distribution"),
        px.box(df, x="gender", y="income_inr", color="gender", title="Income by Gender"),
        px.bar(df.groupby("city")["orders_outside_per_week"].mean(), title="Avg Outside Orders / Week by City"),
        px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week", color="gender", size="work_hours_per_day", title="Commute vs Cooking Frequency"),
        px.pie(df, names="meal_type_pref", title="Meal Type Preference"),
        px.histogram(df, x="dinner_time_hour", color="primary_cook", nbins=24, barmode="overlay", title="Preferred Dinner Hour"),
        px.violin(df, y="non_veg_freq_per_week", x="gender", color="gender", box=True, title="Non-Veg Frequency by Gender"),
        px.sunburst(df, path=["city", "favorite_cuisines"], values="income_inr", title="Cuisine Contribution by City"),
        px.histogram(df, x="work_hours_per_day", color="employment_type", barnorm="percent", title="Work Hours Distribution"),
        px.bar(df.groupby("allergies")["income_inr"].median(), title="Median Income by Allergy Group"),
        px.line(df.sort_values("age"), x="age", y="healthy_importance_rating", title="Health Importance vs Age"),
        px.treemap(df, path=["dietary_goals", "favorite_cuisines"], values="income_inr", title="Diet Goals vs Favourite Cuisines"),
        px.density_heatmap(df, x="age", y="income_inr", title="Age-Income Density"),
        px.scatter_3d(df, x="work_hours_per_day", y="commute_minutes", z="dinners_cooked_per_week", color="gender", title="3-D Lifestyle Cluster"),
        px.area(df.sort_values("age"), x="age", y="orders_outside_per_week", title="Outside Orders vs Age"),
    ]

    for i in range(0, len(charts), 3):
        cols = st.columns(3)
        for fig, col in zip(charts[i : i + 3], cols):
            with col:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(fig.layout.title.text)

# ───────────────────── 2 ▸ CLASSIFICATION ─────────────────────
with tab_clf:
    st.header("Customer Conversion Classification")

    target_col = st.selectbox("Target label", ("subscribe_try", "continue_service", "refer_service"))
    X, y, num_c, cat_c = split_xy(df, target_col)

    # Safe stratification
    strat_ok = y.nunique() == 2 and y.value_counts().min() > 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if strat_ok else None
    )

    k_val = max(1, min(5, len(X_train)))
    models = {
        f"KNN(k={k_val})": KNeighborsClassifier(n_neighbors=k_val),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    roc_fig = go.Figure(layout=go.Layout(title="ROC Curves"))
    result_rows, model_store = [], {}
    for name, est in models.items():
        mdl = build_pipe(est, num_c, cat_c).fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        try:
            y_prob = mdl.predict_proba(X_test)
        except Exception:
            y_prob = None

        result_rows.append(cls_metrics(y_test, y_pred, y_prob).assign(Model=name))
        model_store[name] = mdl, y_pred, y_prob

        # safe ROC add
        safe_roc_add(roc_fig, y_test, y_prob, mdl.classes_, name)

    st.subheader("Performance Metrics")
    st.dataframe(pd.concat(result_rows).set_index("Model"))

    if roc_fig.data:
        st.plotly_chart(roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR"), use_container_width=True)
    else:
        st.info("ROC curves not shown (probabilities unavailable or single-class test split).")

    chosen = st.selectbox("Confusion matrix for:", list(model_store))
    mdl, y_pred_c, _ = model_store[chosen]
    try:
        cm_fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_c).figure_
        st.pyplot(cm_fig)
    except ValueError:
        st.info("Confusion matrix unavailable (test split contains one class).")

    st.markdown("---")
    st.subheader("Batch Prediction")
    upl = st.file_uploader("Upload CSV (data without target)", type="csv")
    if upl:
        new_df = pd.read_csv(upl)
        preds = mdl.predict(new_df)
        new_df[f"pred_{target_col}"] = preds
        buf = BytesIO(); new_df.to_csv(buf, index=False)
        st.download_button("Download predictions", buf.getvalue(), "predictions.csv")
        st.success("Prediction file ready.")

# ───────────────────── 3 ▸ CLUSTERING ─────────────────────
with tab_clust:
    st.header("Persona Discovery – K-means")
    num_all = df.select_dtypes("number").columns.tolist()
    feat = st.multiselect("Numeric features (2-5)", num_all, default=num_all[:4])
    if len(feat) < 2:
        st.warning("Select at least two features."); st.stop()

    k_opt = st.slider("k clusters", 2, 10, 4)
    km = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE).fit(df[feat])
    df["cluster"] = km.labels_
    st.plotly_chart(
        px.scatter(df, x=feat[0], y=feat[1], color="cluster", hover_data=feat, title="Cluster Scatter", color_continuous_scale="Viridis"),
        use_container_width=True,
    )
    st.dataframe(df.groupby("cluster")[feat].mean().round(1))

# ───────────────────── 4 ▸ ASSOCIATION RULES ─────────────────────
with tab_rules:
    st.header("Market-Basket Insights – Apriori")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_pick = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:3])
    sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_val = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if cat_pick:
        tx = df[cat_pick].astype(str).apply(lambda s: s.name + "=" + s)
        basket = pd.get_dummies(tx.stack()).groupby(level=0).sum().astype(bool)
        freq = apriori(basket, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"] >= lift_val]
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10).reset_index(drop=True) if not rules.empty else pd.DataFrame())
    else:
        st.info("Select categorical columns.")

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
        yhat = p.predict(Xr)
        rows.append({"Model": nm, "R²": metrics.r2_score(yr, yhat), "MAE": metrics.mean_absolute_error(yr, yhat)})
        if nm == "Decision Tree":
            imp = p["mdl"].feature_importances_
            ohe_names = p["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_r)
            fi = dict(zip(num_r + ohe_names.tolist(), imp))

    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))
    if fi:
        top = pd.Series(fi).sort_values(ascending=False).head(15)
        st.plotly_chart(px.bar(top, x=top.values, y=top.index, orientation="h", title="Top Feature Importances"), use_container_width=True)

# ───────────────────── 6 ▸ REVENUE FORECAST ─────────────────────
with tab_fcst:
    st.header("12-Month Revenue Forecast by City")

    city_rev = df.assign(revenue=revenue_series(df)).groupby("city")["revenue"].sum()
    mdl_sel = st.selectbox("Model", ["Random Forest", "ARIMA"])
    rows = []
    for city, base in city_rev.items():
        if mdl_sel == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(np.arange(len(city_rev)).reshape(-1, 1), city_rev.values)
            preds = rf.predict(np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1))
        else:
            try:
                ar = ARIMA(city_rev.values, order=(1, 1, 0)).fit()
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
