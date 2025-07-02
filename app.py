"""
Urban Fuel – Consumer Intelligence Hub (Stage 1: Core Shell)
------------------------------------------------------------
Loads the CSV, applies global filters, shows KPI cards, and
sets up empty tabs for later modules.
"""

from __future__ import annotations

# ── Std-lib
from pathlib import Path
from typing import List, Tuple

# ── Third-party
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns

# ── Streamlit global config
try:  # Option removed in recent Streamlit versions
    st.set_option("deprecation.showPyplotGlobalUse", False)
except st.StreamlitAPIException:
    pass

st.set_page_config(
    page_title="Urban Fuel – Consumer Intelligence Hub",
    page_icon="🍱",
    layout="wide",
)

# ── Constants
DATA_PATH = Path("UrbanFuelSyntheticSurvey (1).csv")
RND = 42

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading data…")
def load_data() -> pd.DataFrame:
    """Read the survey CSV and coerce numeric-looking object columns."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    numeric_objects = [
        c
        for c in df.select_dtypes("object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[numeric_objects] = df[numeric_objects].apply(pd.to_numeric, errors="coerce")
    return df


def fmt_inr(x) -> str:
    """Indian-rupee formatting with thousands separator."""
    return "-" if pd.isna(x) else f"₹{int(float(x)):,}"


# ─────────────────────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:8px">
        <img src="https://raw.githubusercontent.com/streamlit/brand/master/logos/2021/streamlit-logo-primary-colormark-lighttext.png"
             width="32">
        <h1 style="display:inline">Urban Fuel – Consumer Intelligence Hub</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
df_raw = load_data()

# ─────────────────────────────────────────────────────────────
# Sidebar filters (global)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global filters")
    sel_city = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    sel_gender = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_lo, inc_hi = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    sel_inc = st.slider("Income (INR)", inc_lo, inc_hi, (inc_lo, inc_hi), 10_000)
    sel_diet = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )
    st.divider()
    dark_mode = st.toggle("🌗 Dark mode")
    px.defaults.template = "plotly_dark" if dark_mode else "plotly_white"

# Apply filters
df = df_raw.copy()
if sel_city:
    df = df[df["city"].isin(sel_city)]
if sel_gender:
    df = df[df["gender"].isin(sel_gender)]
df = df[df["income_inr"].between(*sel_inc)]
if sel_diet:
    df = df[df["dietary_goals"].isin(sel_diet)]

if df.empty:
    st.error("No data remain after filtering. Adjust filters and retry.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Respondents", f"{len(df):,}")
kpi2.metric("Average Age", f"{df['age'].mean():.1f} yrs")
kpi3.metric("Median Income", fmt_inr(df["income_inr"].median()))
kpi4.metric(
    "Health Importance", f"{df['healthy_importance_rating'].mean():.2f} / 5"
)

# ─────────────────────────────────────────────────────────────
# Tab scaffold (content will be filled in subsequent stages)
# ─────────────────────────────────────────────────────────────
tab_exp, tab_cls, tab_clu, tab_rules, tab_reg, tab_fcst = st.tabs(
    [
        "📊 Exploration",
        "🤖 Classification",
        "🧩 Clustering",
        "🔗 Association Rules",
        "📈 Regression",
        "⏳ Forecast",
    ]
)

with tab_exp:
    st.info("**Data-exploration gallery will appear here in Stage 2.**")

with tab_cls:
    st.info("**Classification suite will arrive in Stage 3.**")

with tab_clu:
    st.info("**Clustering lab will be delivered in Stage 4.**")

with tab_rules:
    st.info("**Apriori rule-mining interface coming in Stage 4.**")

with tab_reg:
    st.info("**Regression & feature-importance view scheduled for Stage 5.**")

with tab_fcst:
    st.info("**12-month revenue forecast will be implemented in Stage 5.**")

# Footer
st.markdown(
    "<br><center style='font-size:0.75rem'>© 2025 Urban Fuel Analytics – Stage 1</center>",
    unsafe_allow_html=True,
)
