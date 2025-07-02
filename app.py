# ────────────────────────────────────────────────────────────────────────────────
# app.py  ·  Urban Fuel – Consumer Intelligence Hub  ·  All-Plotly Edition
# ────────────────────────────────────────────────────────────────────────────────
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA

# ────────────────────────────────────────────────────────────────────────────────
# Config & constants
# ────────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")

THEME_PALETTE = px.colors.qualitative.Safe
FORECAST_HORIZON = 12  # months
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    """Load survey CSV and perform basic typing fixes."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    numeric_cols = [
        c
        for c in df.select_dtypes(include="object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def format_inr(x: float | int) -> str:
    """Return INR-formatted string with thousands separator."""
    if pd.isna(x):
        return "-"
    return f"₹{int(x):,}"


def cache_model(func):
    """Decorator to cache heavy model training functions."""

    @st.cache_resource(show_spinner="Training model…")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def split_xy(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Split dataframe into features/target + identify num/cat."""
    y = df[target]
    X = df.drop(columns=[target])

    numerical_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, numerical_cols, categorical_cols


def build_classifier_pipeline(
    estimator, num_cols: List[str], cat_cols: List[str]
) -> Pipeline:
    """Compose preprocessing + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    model = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])
    return model


def metrics_table(y_true, y_pred, y_proba) -> pd.DataFrame:
    """Return classification metric dataframe."""
    return pd.DataFrame(
        {
            "Accuracy": [metrics.accuracy_score(y_true, y_pred)],
            "Precision": [metrics.precision_score(y_true, y_pred, zero_division=0)],
            "Recall": [metrics.recall_score(y_true, y_pred, zero_division=0)],
            "F1-Score": [metrics.f1_score(y_true, y_pred, zero_division=0)],
            "AUC": [
                metrics.roc_auc_score(y_true, y_proba[:, 1])
                if y_proba is not None
                else np.nan
            ],
        }
    ).round(3)


def revenue_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Return per-row revenue approximation.
    Uses willing_to_pay_mealkit_inr if available, else spend_outside_per_meal.
    """
    if "willing_to_pay_mealkit_inr" in df.columns:
        return df["willing_to_pay_mealkit_inr"].fillna(0)
    if "spend_outside_per_meal_inr" in df.columns:
        return df["spend_outside_per_meal_inr"].fillna(0)
    raise KeyError("Revenue columns not found.")


def safe_toast(msg: str, icon: str = "⚠️") -> None:
    """Safely show a toast without crashing in scripted mode."""
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.warning(msg)


# ────────────────────────────────────────────────────────────────────────────────
# Sidebar – global filters
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/<your-repo>",
        "Report a bug": "https://github.com/<your-repo>/issues",
    },
)

df_raw = load_data()

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/simpleicons/simple-icons/develop/icons/bento.svg",
        width=48,
    )
    st.title("Filters")
    city_sel = st.multiselect("City", options=sorted(df_raw["city"].dropna().unique()))
    gender_sel = st.multiselect(
        "Gender", options=sorted(df_raw["gender"].dropna().unique())
    )
    income_min, income_max = (
        int(df_raw["income_inr"].min()),
        int(df_raw["income_inr"].max()),
    )
    income_range = st.slider(
        "Income range (INR)", min_value=income_min, max_value=income_max, value=(income_min, income_max)
    )
    dietary_sel = st.multiselect(
        "Dietary goals", options=sorted(df_raw["dietary_goals"].dropna().unique())
    )
    st.markdown("---")
    st.caption("Global filters apply to all tabs.")

df = df_raw.copy()
if city_sel:
    df = df[df["city"].isin(city_sel)]
if gender_sel:
    df = df[df["gender"].isin(gender_sel)]
df = df[df["income_inr"].between(income_range[0], income_range[1])]
if dietary_sel:
    df = df[df["dietary_goals"].isin(dietary_sel)]

if df.empty:
    safe_toast("No records match selected filters.", icon="❗")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────────
tab_titles = [
    "📊 Data Visualisation",
    "🤖 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression / Impact",
    "⏳ 12-Month Revenue Forecast",
]
(
    tab_viz,
    tab_clf,
    tab_cluster,
    tab_rules,
    tab_reg,
    tab_fcst,
) = st.tabs(tab_titles)

# ────────────────────────────────────────────────────────────────────────────────
# 1. Data Visualisation
# ────────────────────────────────────────────────────────────────────────────────
with tab_viz:
    st.header("Interactive Insights")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Respondents", f"{len(df):,}")
    kpi2.metric("Avg. Age (yrs)", f"{df['age'].mean():.1f}")
    kpi3.metric("Median Income", format_inr(df["income_inr"].median()))
    kpi4.metric(
        "Healthy Eating Importance",
        f"{df['healthy_importance_rating'].mean():.2f}/5",
    )

    # 15 Plotly charts (all-Plotly, no Altair)
    charts = [
        px.histogram(
            df,
            x="income_inr",
            nbins=40,
            title="Income Distribution",
            color_discrete_sequence=[THEME_PALETTE[0]],
        ),
        px.box(
            df,
            x="gender",
            y="income_inr",
            title="Income by Gender",
            color="gender",
            color_discrete_sequence=THEME_PALETTE,
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
            color_discrete_sequence=THEME_PALETTE,
        ),
        px.pie(
            df,
            names="meal_type_pref",
            title="Meal Type Preference",
        ),
        px.histogram(
            df,
            x="dinner_time_hour",
            color="primary_cook",
            barmode="overlay",
            nbins=24,
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
        # ⬇⬇ REPLACED Altair with Plotly histogram ⬇⬇
        px.histogram(
            df,
            x="work_hours_per_day",
            color="employment_type",
            barnorm="percent",
            title="Work Hours Distribution",
            color_discrete_sequence=THEME_PALETTE,
        ),
        # ⬆⬆ REPLACED Altair with Plotly histogram ⬆⬆
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
            df,
            x="age",
            y="income_inr",
            title="Age-Income Density",
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
        for chart, col in zip(charts[i : i + 3], cols):
            with col:
                st.plotly_chart(chart, use_container_width=True)
                st.caption(chart.layout.title.text if chart else "")

    with st.expander("📌 Business Takeaways"):
        st.markdown(
            textwrap.dedent(
                """
                * **Income skews upward** in metro cities, presenting premium pricing headroom.  
                * **Commute length inversely correlates** with home-cooked dinners → convenience positioning.  
                * **Health focus rises with age**, signalling segmented messaging for seniors.  
                * **Non-veg frequency** pockets suggest curated protein-rich meal-kits.
                """
            )
        )

# ────────────────────────────────────────────────────────────────────────────────
# 2. Classification
# (unchanged – omitted here for brevity; keep existing code)
# ────────────────────────────────────────────────────────────────────────────────
# … REMAINDER OF FILE UNCHANGED …
# ────────────────────────────────────────────────────────────────────────────────
