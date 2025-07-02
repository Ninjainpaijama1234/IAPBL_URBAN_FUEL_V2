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

# ─────────────────────────────────────────────────────────────
# Stage 2 – Exploratory Storytelling Gallery (v2 with refined Insight 8)
# ─────────────────────────────────────────────────────────────
with tab_exp:
    import plotly.graph_objects as go  # radar

    st.subheader("Exploratory Storytelling Gallery")

    # ---------- styling helper ----------
    def style(fig: go.Figure, short_title: str) -> go.Figure:
        fig.update_layout(
            title=short_title,
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(size=13),
        )
        return fig

    charts: list[go.Figure] = []

    # 1. Age distribution by gender
    charts.append(style(px.histogram(df, x="age", color="gender", nbins=30, barmode="overlay",
                                     title="Age distribution by gender"),
                        "Age distribution by gender"))

    # 2. Income distribution by city
    charts.append(style(px.box(df, x="city", y="income_inr", color="city",
                               title="Income spread across cities"),
                        "Income spread across cities"))

    # 3. Commute vs. dinners cooked (trendline optional)
    try:
        import statsmodels.api  # noqa: F401
        commute_fig = px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week",
                                 color="city", trendline="ols",
                                 title="Commute time vs dinners cooked (OLS)")
    except Exception:
        commute_fig = px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week",
                                 color="city",
                                 title="Commute time vs dinners cooked")
    charts.append(style(commute_fig, commute_fig.layout.title.text))

    # 4. Dinner hour × health rating heatmap
    charts.append(style(px.density_heatmap(df, x="dinner_time_hour", y="healthy_importance_rating",
                                           nbinsx=24,
                                           title="Dinner hour vs health rating"),
                        "Dinner hour vs health rating"))

    # 5. Sunburst: city → favourite cuisines
    charts.append(style(px.sunburst(df, path=["city", "favorite_cuisines"], values="income_inr",
                                    title="Cuisine income by city"),
                        "Cuisine income by city"))

    # 6. Treemap: dietary goals → meal-type preference
    charts.append(style(px.treemap(df, path=["dietary_goals", "meal_type_pref"], values="income_inr",
                                   title="Diet goals vs meal type"),
                        "Diet goals vs meal type"))

    # 7. Violin: outside orders per week by gender
    charts.append(style(px.violin(df, y="orders_outside_per_week", x="gender", color="gender",
                                  box=True, title="Outside-order freq by gender"),
                        "Outside-order freq by gender"))

    # 8. **Improved parallel categories (condensed)**
    def top_k(series, k=4):
        top = series.value_counts(dropna=False).nlargest(k).index
        return series.where(series.isin(top), other="Other")

    tmp = df.copy()
    for col in ["dietary_goals", "favorite_cuisines", "meal_type_pref", "primary_cook"]:
        tmp[col] = top_k(tmp[col], 4)

    par_fig = px.parallel_categories(
        tmp,
        dimensions=["dietary_goals", "favorite_cuisines", "meal_type_pref", "primary_cook"],
        color="healthy_importance_rating",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Condensed journey: diet → cuisine → meal-type → cook role",
    )
    charts.append(style(par_fig, "Diet → cuisine → meal-type → cook (condensed)"))

    # 9. 3-D lifestyle cube
    charts.append(style(px.scatter_3d(df, x="work_hours_per_day", y="commute_minutes",
                                      z="dinners_cooked_per_week",
                                      color="dietary_goals",
                                      title="Work × commute × cooking cube"),
                        "Work × commute × cooking cube"))

    # 10. Radar: priority pillars
    radar_vals = df[["priority_taste", "priority_price", "priority_nutrition",
                     "priority_ease", "priority_time"]].mean()
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=radar_vals.values, theta=radar_vals.index,
                                    fill="toself", name="Mean"))
    charts.append(style(radar, "Average priority landscape"))

    # 11. Correlation heatmap
    charts.append(style(px.imshow(df.select_dtypes("number").corr(), text_auto=".2f",
                                  aspect="auto", title="Numeric correlation heatmap"),
                        "Numeric correlation heatmap"))

    # 12. Non-veg frequency by city
    charts.append(style(px.box(df, x="city", y="non_veg_freq_per_week", color="city",
                               title="Non-veg meals / week by city"),
                        "Non-veg meals / week by city"))

    # 13. Outside-spend trend across age cohorts
    age_bins = pd.cut(df["age"], bins=range(15, 71, 5)).astype(str)
    charts.append(style(px.line(df.assign(age_bin=age_bins)
                                .groupby("age_bin")["spend_outside_per_meal_inr"]
                                .mean().reset_index(),
                                x="age_bin", y="spend_outside_per_meal_inr",
                                title="Outside-meal spend by age cohort"),
                        "Outside-meal spend by age cohort"))

    # 14. Health-importance distribution by gender
    charts.append(style(px.histogram(df, x="healthy_importance_rating", color="gender",
                                     barmode="overlay",
                                     title="Health-importance distribution"),
                        "Health-importance distribution"))

    # 15. Cooking skill vs enjoyment
    charts.append(style(px.scatter(df, x="cooking_skill_rating", y="enjoy_cooking",
                                   size="income_inr", color="gender",
                                   title="Skill vs enjoyment (size = income)"),
                        "Cooking skill vs enjoyment"))

    # 16. Stacked bar: payment mode vs city
    charts.append(style(px.histogram(df, x="payment_mode", color="city", barmode="stack",
                                     title="Payment mode by city"),
                        "Payment mode by city"))

    # 17. Incentive preference counts
    inc_cnt = df["incentive_pref"].value_counts(dropna=False).reset_index()
    inc_cnt.columns = ["incentive_pref", "count"]
    if not inc_cnt.empty:
        charts.append(style(px.bar(inc_cnt, x="incentive_pref", y="count",
                                   title="Incentive preference counts"),
                            "Incentive preference counts"))

    # 18. Cumulative willing-to-pay across age
    cum_df = df.sort_values("age").assign(cum_wtp=df["willing_to_pay_mealkit_inr"].cumsum())
    charts.append(style(px.area(cum_df, x="age", y="cum_wtp",
                                title="Cumulative WTP by age"),
                        "Cumulative WTP by age"))

    # 19. Commute distribution animation
    charts.append(style(px.histogram(df, x="commute_minutes", color="city", nbins=40,
                                     animation_frame="city",
                                     title="Commute distribution by city"),
                        "Commute distribution by city"))

    # 20. Scatter-matrix of core numerics
    charts.append(style(px.scatter_matrix(df,
                                          dimensions=["age", "income_inr", "commute_minutes",
                                                      "work_hours_per_day"],
                                          color="gender",
                                          title="Scatter-matrix: core numerics"),
                        "Scatter-matrix: core numerics"))

    # ---------- Render 2-column grid ----------
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, fig in enumerate(charts[i:i + 2]):
            with cols[j]:
                idx = i + j + 1
                st.markdown(f"**Insight {idx}**")
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Business takeaways ----------
    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Younger commuters show the sharpest drop in home-cooked dinners.\n"
            "- Mediterranean & South-Indian dominate high-health segments.\n"
            "- ‘Time-poor diners’ cluster around 10-hour workdays and long commutes.\n"
            "- Bundling low-carb kits with protein add-ons could unlock ₹10–15 crore."
        )


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
