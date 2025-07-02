"""
Urban Fuel – Consumer Intelligence Hub  |  Stage 1: Core Shell
----------------------------------------------------------------
• Loads the UrbanFuelSyntheticSurvey CSV.
• Applies global sidebar filters.
• Shows KPI cards.
• Creates empty tabs that later stages will populate.
"""

from __future__ import annotations

# ── Standard library
from pathlib import Path
from typing import List, Tuple

# ── Third-party
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

# ──────────────────── Streamlit global config ───────────────────
# Streamlit ≤1.32 had a pyplot deprecation flag; newer builds removed it.
# The multi-exception guard works on *all* versions.
try:
    from streamlit.errors import StreamlitAPIException
except ModuleNotFoundError:  # ancient Streamlit
    StreamlitAPIException = Exception  # type: ignore

try:
    st.set_option("deprecation.showPyplotGlobalUse", False)
except (StreamlitAPIException, AttributeError, ValueError):
    pass

st.set_page_config(
    page_title="Urban Fuel – Consumer Intelligence Hub",
    page_icon="🍱",
    layout="wide",
)

# ───────────────────────────── Constants ────────────────────────
DATA_PATH = Path("UrbanFuelSyntheticSurvey (1).csv")
RND = 42  # random seed

# ─────────────────────── Helper functions ───────────────────────
@st.cache_data(show_spinner="📂 Loading survey…")
def load_data() -> pd.DataFrame:
    """Read CSV, clean headers, coerce numeric‐strings to numbers."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    numeric_objs = [
        c
        for c in df.select_dtypes("object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[numeric_objs] = df[numeric_objs].apply(pd.to_numeric, errors="coerce")
    return df


def fmt_inr(x) -> str:
    """Indian Rupee formatting with thousands separators."""
    return "-" if pd.isna(x) else f"₹{int(float(x)):,}"


# ──────────────────────────── Header ────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:8px">
        <img src="https://raw.githubusercontent.com/streamlit/brand/master/logos/2021/"
             "streamlit-logo-primary-colormark-lighttext.png" width="32">
        <h1 style="display:inline">Urban Fuel – Consumer Intelligence Hub</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

df_raw = load_data()

# ─────────────────────── Sidebar filters ────────────────────────
with st.sidebar:
    st.header("Global filters")
    sel_city = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    sel_gender = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_min, inc_max = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    sel_inc = st.slider("Income (INR)", inc_min, inc_max, (inc_min, inc_max), 10_000)
    sel_diet = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )
    st.divider()
    dark_mode = st.toggle("🌗 Dark mode")

# honour dark-mode template
px.defaults.template = "plotly_dark" if dark_mode else "plotly_white"
sns.set_palette("Set2")

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
    st.error("⚠️ No rows match your filters. Please broaden your selection.")
    st.stop()

# ───────────────────────────── KPI cards ─────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Respondents", f"{len(df):,}")
k2.metric("Average Age", f"{df['age'].mean():.1f} yrs")
k3.metric("Median Income", fmt_inr(df["income_inr"].median()))
k4.metric("Health Importance", f"{df['healthy_importance_rating'].mean():.2f} / 5")

# ─────────────────────────── Tab scaffold ────────────────────────
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
    st.info("**Apriori rule-mining UI coming in Stage 4.**")

with tab_reg:
    st.info("**Regression & feature-importance view scheduled for Stage 5.**")

with tab_fcst:
    st.info("**12-month revenue forecast will be implemented in Stage 5.**")

# ──────────────────────────── Footer ─────────────────────────────
st.markdown(
    "<br><center style='font-size:0.75rem'>© 2025 Urban Fuel Analytics — Stage 1</center>",
    unsafe_allow_html=True,
)
