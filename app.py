"""
Urban Fuel – Consumer Intelligence Hub
All-Plotly, production-ready Streamlit dashboard.
----------------------------------------------------------------
Tabs
1. 📊 Data Visualisation          – 15 interactive charts + KPIs
2. 🤖 Classification              – 4 algorithms, metrics, ROC, CM
3. 🧩 Clustering                  – K-means elbow + persona explorer
4. 🔗 Association Rules           – Apriori with custom thresholds
5. 📈 Regression / Feature Impact – 4 regressors + FI bar chart
6. ⏳ 12-Month Revenue Forecast   – RF or ARIMA by city
----------------------------------------------------------------
Author : <your name>
"""

from __future__ import annotations

import logging
import pathlib
import textwrap
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

# ────────────────────────────── CONFIG ──────────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")
THEME = px.colors.qualitative.Safe
FORECAST_HORIZON = 12  # months
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────── HELPER FUNCTIONS ───────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_PATH)
    df_.columns = (
        df_.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    # Coerce numeric-looking object columns
    num_like = [
        c
        for c in df_.select_dtypes(include="object").columns
        if df_[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df_[num_like] = df_[num_like].apply(pd.to_numeric, errors="coerce")
    return df_


def fmt_inr(x: float | int) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def safe_toast(msg: str, icon: str = "⚠️"):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.warning(msg)


def split_xy(
    data: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y_ = data[target]
    X_ = data.drop(columns=[target])
    num_cols_ = X_.select_dtypes(include="number").columns.tolist()
    cat_cols_ = X_.select_dtypes(exclude="number").columns.tolist()
    return X_, y_, num_cols_, cat_cols_


def build_pipe(estimator, num_cols, cat_cols) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("prep", prep), ("mdl", estimator)])


def classification_metrics(y_true, y_pred, y_prob=None) -> pd.DataFrame:
    """Handles binary or multi-class gracefully."""
    if y_true.nunique() == 2:
        pr = metrics.precision_score(y_true, y_pred, zero_division=0)
        rc = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        auc = metrics.roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else np.nan
    else:
        pr = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rc = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc = np.nan
    acc = metrics.accuracy_score(y_true, y_pred)
    return pd.DataFrame(
        {
            "Accuracy": [acc],
            "Precision": [pr],
            "Recall": [rc],
            "F1": [f1],
            "AUC": [auc],
        }
    ).round(3)


def revenue_series(df_: pd.DataFrame) -> pd.Series:
    if "willing_to_pay_mealkit_inr" in df_.columns:
        return df_["willing_to_pay_mealkit_inr"].fillna(0)
    return df_["spend_outside_per_meal_inr"].fillna(0)


# ─────────────────────────── STREAMLIT LAYOUT ───────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

df_raw = load_data()

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.title("Filters")
    city_f = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_f = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_min, inc_max = (
        int(df_raw["income_inr"].min()),
        int(df_raw["income_inr"].max()),
    )
    inc_range = st.slider(
        "Income range (INR)",
        inc_min,
        inc_max,
        (inc_min, inc_max),
        step=10000,
    )
    diet_f = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )
    st.caption("All filters apply to every tab.")

df = df_raw.copy()
if city_f:
    df = df[df["city"].isin(city_f)]
if gender_f:
    df = df[df["gender"].isin(gender_f)]
df = df[df["income_inr"].between(*inc_range)]
if diet_f:
    df = df[df["dietary_goals"].isin(diet_f)]

if df.empty:
    safe_toast("No records match selected filters.", "❗")
    st.stop()

# ---------- Tabs ----------
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

# ───────────────────── 1 ▸ DATA VISUALISATION ─────────────────────
with tab_viz:
    st.header("Interactive Insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Avg. Age", f"{df['age'].mean():.1f}")
    c3.metric("Median Income", fmt_inr(df["income_inr"].median()))
    c4.metric("Health Importance", f"{df['healthy_importance_rating'].mean():.2f}/5")

    charts: List[go.Figure] = [
        px.histogram(
            df,
            x="income_inr",
            nbins=40,
            title="Income Distribution",
            color_discrete_sequence=[THEME[0]],
        ),
        px.box(
            df,
            x="gender",
            y="income_inr",
            color="gender",
            title="Income by Gender",
            color_discrete_sequence=THEME,
        ),
        px.bar(
            df.groupby("city")["orders_outside_per_week"].mean().sort_values(),
            title="Avg Outside Orders / Week by City",
        ),
        px.scatter(
            df,
            x="commute_minutes",
            y="dinners_cooked_per_week",
            color="gender",
            size="work_hours_per_day",
            title="Commute vs Cooking Frequency",
            color_discrete_sequence=THEME,
        ),
        px.pie(df, names="meal_type_pref", title="Meal-Type Preference"),
        px.histogram(
            df,
            x="dinner_time_hour",
            color="primary_cook",
            nbins=24,
            barmode="overlay",
            title="Preferred Dinner Time",
        ),
        px.violin(
            df,
            y="non_veg_freq_per_week",
            x="gender",
            color="gender",
            box=True,
            title="Non-Veg Frequency by Gender",
        ),
        px.sunburst(
            df,
            path=["city", "favorite_cuisines"],
            values="income_inr",
            title="Cuisines by City & Income Contribution",
        ),
        px.histogram(
            df,
            x="work_hours_per_day",
            color="employment_type",
            barnorm="percent",
            title="Work Hours Distribution",
            color_discrete_sequence=THEME,
        ),
        px.bar(
            df.groupby("allergies")["income_inr"].median(),
            title="Median Income by Allergy Group",
        ),
        px.line(
            df.sort_values("age"),
            x="age",
            y="healthy_importance_rating",
            title="Health Importance Across Age",
        ),
        px.treemap(
            df,
            path=["dietary_goals", "favorite_cuisines"],
            values="income_inr",
            title="Diet Goals vs Favourite Cuisines",
        ),
        px.density_heatmap(
            df, x="age", y="income_inr", title="Age-Income Density"
        ),
        px.scatter_3d(
            df,
            x="work_hours_per_day",
            y="commute_minutes",
            z="dinners_cooked_per_week",
            color="gender",
            title="3-D Lifestyle Cluster",
        ),
        px.area(
            df.sort_values("age"),
            x="age",
            y="orders_outside_per_week",
            title="Outside Orders vs Age",
        ),
    ]

    for i in range(0, len(charts), 3):
        cols = st.columns(3)
        for fig, col in zip(charts[i : i + 3], cols):
            with col:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(fig.layout.title.text)

    with st.expander("📌 Business Takeaways"):
        st.markdown(
            textwrap.dedent(
                """
                • **Metro cities skew richer** → premium pricing opportunity.  
                • Longer commutes ↔ fewer home-cooked dinners → convenience value prop.  
                • Health focus rises with age, supporting age-tailored messaging.  
                • Discrete non-veg clusters suggest protein-rich kit variants.
                """
            )
        )

# ───────────────────── 2 ▸ CLASSIFICATION ─────────────────────
with tab_clf:
    st.header("Customer Conversion Classification")

    target_col = st.selectbox(
        "Binary target column",
        ("subscribe_try", "continue_service", "refer_service"),
    )
    X, y, num_cols, cat_cols = split_xy(df, target_col)

    # Safe stratification
    class_counts = y.value_counts()
    stratify_ok = y.nunique() == 2 and class_counts.min() > 1
    if not stratify_ok and y.nunique() == 2:
        st.info(
            f"Stratification disabled (minority class size = {class_counts.min()}).",
            icon="ℹ️",
        )

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y if stratify_ok else None,
    )

    algos = {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: Dict[str, Dict] = {}
    roc_fig = go.Figure(layout=go.Layout(title="ROC Curve"))

    for name, est in algos.items():

        @st.cache_resource(show_spinner=f"Training {name}…")
        def train(model):
            pipe = build_pipe(model, num_cols, cat_cols)
            pipe.fit(X_tr, y_tr)
            return pipe

        model = train(est)
        y_pred = model.predict(X_te)
        try:
            y_prob = model.predict_proba(X_te)
        except AttributeError:
            y_prob = None

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": classification_metrics(y_te, y_pred, y_prob),
        }

        # ROC for binary only
        if y.nunique() == 2 and y_prob is not None:
            fpr, tpr, _ = metrics.roc_curve(y_te, y_prob[:, 1])
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))

    st.subheader("Performance Metrics")
    st.dataframe(
        pd.concat(
            [v["metrics"].assign(Model=k) for k, v in results.items()]
        ).set_index("Model")
    )

    if y.nunique() == 2 and any(v["y_prob"] is not None for v in results.values()):
        st.plotly_chart(
            roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR"),
            use_container_width=True,
        )

    chosen = st.selectbox("Confusion matrix for:", list(results.keys()))
    cm_fig = ConfusionMatrixDisplay.from_predictions(
        y_te, results[chosen]["y_pred"]
    ).figure_
    st.pyplot(cm_fig)

    # ---------- Batch prediction ----------
    st.markdown("---")
    st.subheader("Batch Prediction")
    upload = st.file_uploader("Upload CSV (no target)", type="csv")
    if upload:
        new_df = pd.read_csv(upload)
        try:
            preds = results[chosen]["model"].predict(new_df)
            new_df[f"pred_{target_col}"] = preds
            buf = BytesIO()
            new_df.to_csv(buf, index=False)
            st.download_button("Download predictions", buf.getvalue(), "predictions.csv")
            st.success("✅ Prediction file ready.")
        except Exception as exc:
            safe_toast(f"Prediction failed: {exc}")

# ───────────────────── 3 ▸ CLUSTERING ─────────────────────
with tab_clust:
    st.header("Persona Discovery – K-means")

    numeric_all = df.select_dtypes(include="number").columns.tolist()
    features = st.multiselect(
        "Numeric features (2-5)", numeric_all, default=numeric_all[:4]
    )
    if len(features) < 2:
        st.warning("Pick at least two features.")
        st.stop()

    k_val = st.slider("k (clusters)", 2, 10, 4)
    inertia = [
        KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(df[features]).inertia_
        for k in range(2, 11)
    ]
    st.plotly_chart(
        px.line(x=list(range(2, 11)), y=inertia, markers=True, title="Elbow Curve"),
        use_container_width=True,
    )

    km = KMeans(n_clusters=k_val, random_state=RANDOM_STATE).fit(df[features])
    df["cluster"] = km.labels_

    st.plotly_chart(
        px.scatter(
            df,
            x=features[0],
            y=features[1],
            color="cluster",
            hover_data=features,
            title="Cluster Scatter",
            color_continuous_scale="Viridis",
        ),
        use_container_width=True,
    )

    st.subheader("Cluster Personas (mean values)")
    st.dataframe(df.groupby("cluster")[features].mean().round(1))

    csv_buf = BytesIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download clustered data", csv_buf.getvalue(), "clustered.csv")

# ───────────────────── 4 ▸ ASSOCIATION RULES ─────────────────────
with tab_rules:
    st.header("Market-Basket Insights (Apriori)")

    cat_cols_all = df.select_dtypes(exclude="number").columns.tolist()
    cats = st.multiselect(
        "Choose up to 3 categorical columns", cat_cols_all, default=cat_cols_all[:3]
    )
    support = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    confidence = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_val = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if not cats:
        st.info("Select categorical columns to generate rules.")
    else:
        # Transaction one-hot encoding
        tx = (
            df[cats]
            .astype(str)
            .apply(lambda col: "___" + col.name + "__" + col.str.strip())
        )
        basket = pd.get_dummies(tx.stack()).groupby(level=0).sum().astype(bool)

        freq = apriori(basket, min_support=support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=confidence)
        rules = rules[rules["lift"] >= lift_val]

        if rules.empty:
            st.info("No rules satisfy thresholds.")
        else:
            st.dataframe(
                rules.sort_values("confidence", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

# ───────────────────── 5 ▸ REGRESSION / IMPACT ─────────────────────
with tab_reg:
    st.header("Drivers of Continuous Outcomes")

    num_targets = df.select_dtypes(include="number").columns.tolist()
    target_reg = st.selectbox("Numeric target", num_targets)
    Xr, yr, num_r, cat_r = split_xy(df, target_reg)

    reg_algos = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    }

    rows, fi_dict = [], {}
    for name, reg in reg_algos.items():

        @st.cache_resource(show_spinner=f"Training {name}…")
        def train_reg(est):
            return build_pipe(est, num_r, cat_r)

        pipe = train_reg(reg)
        pipe.fit(Xr, yr)
        preds = pipe.predict(Xr)
        rows.append(
            {
                "Model": name,
                "R²": round(metrics.r2_score(yr, preds), 3),
                "MAE": round(metrics.mean_absolute_error(yr, preds), 1),
            }
        )

        if name == "Decision Tree":
            imp = pipe["mdl"].feature_importances_
            ohe_names = (
                pipe["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_r)
            )
            fi_dict = dict(zip(num_r + ohe_names.tolist(), imp))

    st.dataframe(pd.DataFrame(rows).set_index("Model"))

    if fi_dict:
        top_imp = pd.Series(fi_dict).sort_values(ascending=False).head(15)
        st.plotly_chart(
            px.bar(
                top_imp,
                x=top_imp.values,
                y=top_imp.index,
                orientation="h",
                title="Top Feature Importances (Decision Tree)",
            ),
            use_container_width=True,
        )

# ───────────────────── 6 ▸ 12-MONTH REVENUE FORECAST ─────────────────────
with tab_fcst:
    st.header("Revenue Forecast by City – Next 12 Months")

    rev = revenue_series(df)
    city_rev = df.assign(revenue=rev).groupby("city")["revenue"].sum()

    model_choice = st.selectbox("Forecast model", ["Random Forest", "ARIMA"])
    proj_rows = []

    for city, base in city_rev.items():
        if model_choice == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(
                np.arange(len(city_rev)).reshape(-1, 1), city_rev.values
            )
            preds = rf.predict(
                np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1)
            )
        else:
            try:
                ar = ARIMA(np.array(city_rev), order=(1, 1, 0)).fit()
                preds = ar.forecast(FORECAST_HORIZON)
            except Exception:
                preds = np.full(FORECAST_HORIZON, base)

        for i in range(FORECAST_HORIZON):
            month = (datetime.now() + relativedelta(months=i + 1)).strftime("%Y-%m")
            proj_rows.append({"city": city, "month": month, "forecast": preds[i]})

    proj_df = pd.DataFrame(proj_rows)
    st.dataframe(
        proj_df.pivot(index="month", columns="city", values="forecast")
        .round(0)
        .style.format(fmt_inr)
    )

    st.plotly_chart(
        px.line(
            proj_df,
            x="month",
            y="forecast",
            color="city",
            title="Forecasted Monthly Revenue",
        ),
        use_container_width=True,
    )

# ---------- Footer ----------
st.markdown(
    "<br><center>© 2025 Urban Fuel Analytics · Built with Streamlit</center>",
    unsafe_allow_html=True,
)
