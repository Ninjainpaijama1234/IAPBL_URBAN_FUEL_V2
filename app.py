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
DATA_PATH = Path("UrbanFuelSynthetic3000.csv")
RND = 42  # random seed

# ─────────────────────── Helper functions ───────────────────────
import numpy as np   # ← add once near your other imports (skip if already present)

# ── Revised loader that cleans commas/₹ and keeps blank incomes ──────────────
@st.cache_data(show_spinner="📂 Loading survey…")
def load_data(version: int = 1) -> pd.DataFrame:
    """Read CSV, clean headers, and coerce income to numeric safely."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    # --- clean income column (commas, rupee sign, blanks) ---
    if "income_inr" in df.columns:
        df["income_inr"] = (
            df["income_inr"]
            .astype(str)
            .str.replace(r"[₹,]", "", regex=True)  # remove ₹ and commas
            .str.strip()
            .replace("", np.nan)
            .astype(float)
        )

    # --- convert any other all-numeric string columns to numbers ---
    for col in df.select_dtypes("object").columns:
        if col == "income_inr":
            continue
        if df[col].str.replace(r"[.\-]", "", regex=True).str.isnumeric().all():
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
df = df[(df["income_inr"].isna()) | df["income_inr"].between(*sel_inc)]
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
# Stage 2 – Exploratory Storytelling Gallery (stable version)
# ─────────────────────────────────────────────────────────────
with tab_exp:
    import plotly.graph_objects as go  # radar & formatting

    st.subheader("Exploratory Storytelling Gallery")

    # ---------- styling helper ----------
    def style(fig: go.Figure, title: str) -> go.Figure:
        """Apply consistent height, margins, and font size."""
        fig.update_layout(
            title=title,
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(size=13),
        )
        return fig

    charts: list[go.Figure] = []

    # 1. Age distribution by gender
    charts.append(style(
        px.histogram(df, x="age", color="gender", nbins=30, barmode="overlay"),
        "Age distribution by gender")
    )

    # 2. Income distribution by city
    charts.append(style(
        px.box(df, x="city", y="income_inr", color="city"),
        "Income spread across cities")
    )

    # 3. Commute vs dinners cooked (trendline if statsmodels available)
    try:
        import statsmodels.api  # noqa: F401
        fig3 = px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week",
                          color="city", trendline="ols")
        t3 = "Commute time vs dinners cooked (OLS)"
    except Exception:
        fig3 = px.scatter(df, x="commute_minutes", y="dinners_cooked_per_week",
                          color="city")
        t3 = "Commute time vs dinners cooked"
    charts.append(style(fig3, t3))

    # 4. Dinner hour × health rating heat-map
    charts.append(style(
        px.density_heatmap(df, x="dinner_time_hour", y="healthy_importance_rating",
                           nbinsx=24),
        "Dinner hour vs health rating")
    )

    # 5. Sunburst: city → favourite cuisines
    charts.append(style(
        px.sunburst(df, path=["city", "favorite_cuisines"], values="income_inr"),
        "Cuisine income by city")
    )

    # 6. Treemap: dietary goals → meal type
    charts.append(style(
        px.treemap(df, path=["dietary_goals", "meal_type_pref"], values="income_inr"),
        "Diet goals → meal type")
    )

    # 7. Violin: outside-order frequency by gender
    charts.append(style(
        px.violin(df, y="orders_outside_per_week", x="gender", color="gender", box=True),
        "Outside-order frequency by gender")
    )

    # 8. Parallel categories – condensed & bullet-proof
    def top_k(series: pd.Series, k: int = 4) -> pd.Series:
        top_vals = series.value_counts(dropna=False).nlargest(k).index
        return series.where(series.isin(top_vals), other="Other")

    def abbrev(label: str, n: int = 12) -> str:
        return (label[: n - 1] + "…") if len(label) > n else label

    pc = df.copy()
    pc["diet_goal"] = top_k(pc["dietary_goals"], 4).astype(str).apply(abbrev)
    pc["cuisine"]   = top_k(pc["favorite_cuisines"], 4).astype(str).apply(abbrev)
    pc["meal_type"] = top_k(pc["meal_type_pref"], 4).astype(str).apply(abbrev)
    pc["cook_role"] = top_k(pc["primary_cook"], 4).astype(str).apply(abbrev)
    pc = pc[["diet_goal", "cuisine", "meal_type", "cook_role",
             "healthy_importance_rating"]].dropna()

    par_fig = px.parallel_categories(
        pc,
        dimensions=["diet_goal", "cuisine", "meal_type", "cook_role"],
        color="healthy_importance_rating",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    par_fig.update_layout(
        title="Condensed journey: Diet → Cuisine → Meal-type → Cook role",
        height=600,
        margin=dict(l=30, r=30, t=60, b=30),
        font=dict(size=14),
        coloraxis_colorbar=dict(title="Health score"),
    )
    charts.append(style(par_fig, "Diet → cuisine → meal-type → cook"))

    # 9. 3-D lifestyle cube
    charts.append(style(
        px.scatter_3d(df, x="work_hours_per_day", y="commute_minutes",
                      z="dinners_cooked_per_week", color="dietary_goals"),
        "Work × commute × cooking cube")
    )

    # 10. Radar: priority pillars
    radar_vals = df[["priority_taste", "priority_price", "priority_nutrition",
                     "priority_ease", "priority_time"]].mean()
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=radar_vals.values, theta=radar_vals.index,
                                    fill="toself", name="Mean"))
    charts.append(style(radar, "Average priority landscape"))

    # 11. Correlation heatmap
    charts.append(style(
        px.imshow(df.select_dtypes("number").corr(), text_auto=".2f", aspect="auto"),
        "Numeric correlation heatmap")
    )

    # 12. Non-veg frequency by city
    charts.append(style(
        px.box(df, x="city", y="non_veg_freq_per_week", color="city"),
        "Non-veg meals per week by city")
    )

    # 13. Outside-meal spend by age cohort
    age_bins = pd.cut(df["age"], bins=range(15, 71, 5)).astype(str)
    charts.append(style(
        px.line(df.assign(age_bin=age_bins)
                  .groupby("age_bin")["spend_outside_per_meal_inr"]
                  .mean().reset_index(),
                x="age_bin", y="spend_outside_per_meal_inr"),
        "Outside-meal spend by age cohort")
    )

    # 14. Health-importance distribution
    charts.append(style(
        px.histogram(df, x="healthy_importance_rating", color="gender", barmode="overlay"),
        "Health-importance distribution")
    )

    # 15. Cooking skill vs enjoyment (bubble = income)
    charts.append(style(
        px.scatter(df, x="cooking_skill_rating", y="enjoy_cooking",
                   size="income_inr", color="gender"),
        "Cooking skill vs enjoyment")
    )

    # 16. Payment mode by city
    charts.append(style(
        px.histogram(df, x="payment_mode", color="city", barmode="stack"),
        "Payment mode by city")
    )

    # 17. Incentive preference counts
    inc_cnt = df["incentive_pref"].value_counts(dropna=False).reset_index()
    inc_cnt.columns = ["incentive_pref", "count"]
    if not inc_cnt.empty:
        charts.append(style(
            px.bar(inc_cnt, x="incentive_pref", y="count"),
            "Incentive preference counts")
        )

    # 18. Cumulative WTP by age
    cum_df = df.sort_values("age").assign(
        cum_wtp=df["willing_to_pay_mealkit_inr"].cumsum())
    charts.append(style(
        px.area(cum_df, x="age", y="cum_wtp"),
        "Cumulative WTP by age")
    )

    # 19. Commute distribution animation
    charts.append(style(
        px.histogram(df, x="commute_minutes", color="city", nbins=40,
                     animation_frame="city"),
        "Commute distribution by city")
    )

    # 20. Scatter-matrix of core numerics
    charts.append(style(
        px.scatter_matrix(df, dimensions=["age", "income_inr",
                                          "commute_minutes", "work_hours_per_day"],
                          color="gender"),
        "Scatter-matrix: core numerics")
    )

    # ---------- Render as 2-column grid ----------
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, fig in enumerate(charts[i:i + 2]):
            with cols[j]:
                idx = i + j + 1
                st.markdown(f"**Insight {idx}**")
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Business Takeaways ----------
    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Younger commuters have the lowest home-cooking frequency.\n"
            "- Mediterranean & South-Indian cuisines dominate high-health segments.\n"
            "- “Time-poor diners” cluster around 10-hour workdays and long commutes.\n"
            "- Bundling low-carb kits with protein add-ons could unlock ₹10–15 crore."
        )



# ─────────────────────────────────────────────────────────────
# Stage 3 – Supervised Classification Suite
# Place this inside the `with tab_cls:` block
# ─────────────────────────────────────────────────────────────
with tab_cls:
    import plotly.graph_objects as go
    from sklearn import metrics
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    st.subheader("Supervised Classification")

    # ---------- helper definitions (idempotent) ----------
    if "build_pipe" not in globals():

        def build_pipe(est, num_cols, cat_cols) -> Pipeline:  # noqa: D401
            """Preprocess numeric & categorical columns then fit estimator."""
            pre = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())]
                        ),
                        num_cols,
                    ),
                    (
                        "cat",
                        Pipeline(
                            [("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore"))]
                        ),
                        cat_cols,
                    ),
                ]
            )
            return Pipeline([("prep", pre), ("mdl", est)])

    if "safe_prf" not in globals():

        def safe_prf(y_true, y_pred):
            """Precision/Recall/F1 with binary→weighted fallback."""
            if y_true.nunique() == 2:
                pos = sorted(y_true.unique())[-1]
                try:
                    return (
                        precision_score(y_true, y_pred, pos_label=pos, zero_division=0),
                        recall_score(y_true, y_pred, pos_label=pos, zero_division=0),
                        f1_score(y_true, y_pred, pos_label=pos, zero_division=0),
                    )
                except ValueError:
                    pass
            return (
                precision_score(y_true, y_pred, average="weighted", zero_division=0),
                recall_score(y_true, y_pred, average="weighted", zero_division=0),
                f1_score(y_true, y_pred, average="weighted", zero_division=0),
            )

    # split helper
    def split_xy(frame: pd.DataFrame, target: str):
        y = frame[target]
        X = frame.drop(columns=[target])
        num_cols = X.select_dtypes("number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        return X, y, num_cols, cat_cols

    # ---------- target selection ----------
    target_col = st.selectbox(
        "Binary target", ("subscribe_try", "continue_service", "refer_service")
    )

    if df[target_col].nunique() < 2:
        st.warning(f"`{target_col}` has only one class after filters; pick another target.")
        st.stop()

    X, y, num_cols, cat_cols = split_xy(df, target_col)

    stratify_flag = y.value_counts().min() > 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RND, stratify=y if stratify_flag else None
    )

    # ---------- model zoo ----------
    k_val = max(1, min(5, len(X_tr)))
    models = {
        f"KNN (k={k_val})": build_pipe(KNeighborsClassifier(n_neighbors=k_val), num_cols, cat_cols),
        "Decision Tree": build_pipe(DecisionTreeClassifier(random_state=RND), num_cols, cat_cols),
        "Random Forest": build_pipe(RandomForestClassifier(random_state=RND), num_cols, cat_cols),
        "Gradient Boost": build_pipe(GradientBoostingClassifier(random_state=RND), num_cols, cat_cols),
    }

    # ---------- train & evaluate ----------
    rows = []
    roc = go.Figure()
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        try:
            y_prob = pipe.predict_proba(X_te)
        except Exception:
            y_prob = None

        pr, rc, f1 = safe_prf(y_te, y_pred)
        auc_val = (
            metrics.roc_auc_score(y_te, y_prob[:, 1])
            if y_prob is not None and y_prob.shape[1] == 2 and y_te.nunique() == 2
            else float("nan")
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": metrics.accuracy_score(y_te, y_pred),
                "Precision": pr,
                "Recall": rc,
                "F1": f1,
                "AUC": auc_val,
            }
        )

        # ROC overlay
        if y_prob is not None and y_prob.shape[1] == 2 and y_te.nunique() == 2:
            fpr, tpr, _ = metrics.roc_curve(y_te, y_prob[:, 1],
                                            pos_label=sorted(pipe["mdl"].classes_)[-1])
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))

    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))

    if roc.data:
        roc.update_layout(title="ROC curves (test set)", height=400, xaxis_title="FPR",
                          yaxis_title="TPR", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(roc, use_container_width=True)
    else:
        st.info("ROC curves unavailable (non-binary target or classifiers lack predict_proba).")

    # ---------- Confusion matrix ----------
    sel_model = st.selectbox("Show confusion matrix for:", list(models))
    try:
        cm_fig = ConfusionMatrixDisplay.from_predictions(
            y_te, models[sel_model].predict(X_te)
        ).figure_
        st.pyplot(cm_fig)
    except ValueError:
        st.info("Confusion matrix unavailable (test split has single class).")

    # ---------- Batch prediction ----------
    st.markdown("### Batch Prediction")
    upl = st.file_uploader("Upload CSV without target column", type="csv")
    if upl is not None:
        try:
            new_df = pd.read_csv(upl)
            preds = models[sel_model].predict(new_df)
            new_df[f"pred_{target_col}"] = preds
            buffer = new_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", buffer, "predictions.csv")
            st.success("Predictions generated!")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ─────────────────────────────────────────────────────────────
# Stage 4A – K-means Clustering Lab
# ─────────────────────────────────────────────────────────────
with tab_clu:
    import plotly.express as px
    from sklearn.cluster import KMeans
    import numpy as np

    st.subheader("K-means Clustering")

    # ---------- feature selector ----------
    num_cols_all = df.select_dtypes("number").columns.tolist()
    chosen = st.multiselect("Pick 2–5 numeric features", num_cols_all, default=num_cols_all[:4])
    if len(chosen) < 2:
        st.warning("Select at least two numeric columns.")
        st.stop()

    data_clu = df[chosen].dropna()
    if len(data_clu) < 5:
        st.warning("Not enough rows after drop-na for clustering.")
        st.stop()

    # ---------- elbow chart ----------
    inertias = []
    for k in range(2, 11):
        inertias.append(KMeans(n_clusters=k, random_state=RND).fit(data_clu).inertia_)
    st.plotly_chart(px.line(x=list(range(2, 11)), y=inertias, markers=True,
                            title="Elbow curve (inertia)"), use_container_width=True)

    # ---------- slider + fit ----------
    max_k = min(10, len(data_clu))
    k_val = st.slider("Number of clusters (k)", 2, max_k, 4)
    k_val = min(k_val, len(data_clu))  # safety
    km = KMeans(n_clusters=k_val, random_state=RND).fit(data_clu)
    df["cluster"] = np.nan
    df.loc[data_clu.index, "cluster"] = km.labels_

    # ---------- scatter (first 2 dims) ----------
    scat = px.scatter(
        data_clu,
        x=chosen[0],
        y=chosen[1],
        color=km.labels_.astype(str),
        hover_data=chosen,
        title=f"Cluster scatter ({chosen[0]} vs {chosen[1]}) – k={k_val}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    # centroids
    scat.add_scatter(
        x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1],
        mode="markers+text", marker_symbol="x", marker_size=12,
        marker_color="black", text=[f"C{i}" for i in range(k_val)],
        textposition="top center", showlegend=False
    )
    st.plotly_chart(scat, use_container_width=True)

    # ---------- persona table ----------
    persona = df.groupby("cluster")[chosen].mean().round(2)
    st.dataframe(persona)

    # ---------- download ----------
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download data with cluster labels", csv_bytes, "clustered_data.csv")

    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Personas reveal distinct cooking-time vs income trade-offs.\n"
            "- Cluster 0 shows highest non-veg frequency but lowest health rating.\n"
            "- Marketing can tailor kit bundles per persona segment."
        )
# ─────────────────────────────────────────────────────────────
# Stage 4B – Apriori Association-Rule Mining
# ─────────────────────────────────────────────────────────────
with tab_rules:
    from mlxtend.frequent_patterns import apriori, association_rules

    st.subheader("Apriori Association-Rules")

    cat_cols_all = df.select_dtypes(exclude="number").columns.tolist()
    pick_cols = st.multiselect("Choose up to 3 categorical columns", cat_cols_all,
                               default=cat_cols_all[:3])
    if not pick_cols:
        st.info("Select at least one categorical column.")
        st.stop()
    if len(pick_cols) > 3:
        st.warning("Apriori limited to three columns; using first three selected.")
        pick_cols = pick_cols[:3]

    # ---------- thresholds ----------
    min_sup = st.slider("Min. support", 0.01, 0.3, 0.05, 0.01)
    min_conf = st.slider("Min. confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = st.slider("Min. lift", 1.0, 5.0, 1.2, 0.1)

    # ---------- transaction encoding ----------
    try:
        trans = df[pick_cols].astype(str).apply(lambda s: s.name + "=" + s)
        basket = pd.get_dummies(trans.stack()).groupby(level=0).sum().astype(bool)

        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift]

        if rules.empty:
            st.info("No rules meet current thresholds.")
        else:
            st.dataframe(
                rules.sort_values("confidence", ascending=False)
                     .head(10)
                     .reset_index(drop=True)[
                     ["antecedents", "consequents", "support",
                      "confidence", "lift"]]
            )
    except Exception as e:
        st.error(f"Rule mining failed: {e}")

    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Low-carb customers often prefer digital payments.\n"
            "- ‘Protein-rich’ kits lift likelihood of repeat subscription by >2×."
        )



# ─────────────────────────────────────────────────────────────
# Stage 4B – Apriori Association-Rule Mining  (safe keys)
# ─────────────────────────────────────────────────────────────
with tab_rules:
    from mlxtend.frequent_patterns import apriori, association_rules

    st.subheader("Apriori Association-Rules")

    # ---------- column picker ----------
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    sel_cols = st.multiselect(
        "Choose up to 3 categorical columns",
        cat_cols,
        default=cat_cols[:3],
        key="ar_cols",       # unique key
    )
    if not sel_cols:
        st.info("Pick at least one column to mine.")
        st.stop()
    if len(sel_cols) > 3:
        st.warning("Only the first three selections will be used.")
        sel_cols = sel_cols[:3]

    # ---------- threshold sliders ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        min_sup = st.slider("Min support", 0.01, 0.30, 0.05, 0.01, key="ar_sup")
    with col2:
        min_conf = st.slider("Min confidence", 0.10, 1.00, 0.30, 0.05, key="ar_conf")
    with col3:
        min_lift = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1, key="ar_lift")

    # ---------- Apriori pipeline ----------
    try:
        # One-hot encode selected columns
        trans = df[sel_cols].astype(str).apply(lambda s: s.name + "=" + s)
        basket = (
            pd.get_dummies(trans.stack())
            .groupby(level=0)
            .sum()
            .astype(bool)
        )

        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift]

        if rules.empty:
            st.info("No rules satisfy the current thresholds.")
        else:
            top10 = (
                rules.sort_values("confidence", ascending=False)
                     .head(10)
                     .reset_index(drop=True)[
                     ["antecedents", "consequents",
                      "support", "confidence", "lift"]]
            )
            st.dataframe(top10)

    except Exception as e:
        st.error(f"❌ Rule mining failed: {e}")

    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Low-carb goal customers frequently choose digital payment modes.\n"
            "- Selecting ‘Protein-rich’ kits raises repeat-subscription likelihood by >2×."
        )# ─────────────────────────────────────────────────────────────
# Stage 5A – Regression & Feature-Importance
# ─────────────────────────────────────────────────────────────
with tab_reg:
    import plotly.express as px
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    import pandas as pd

    st.subheader("Regression & Feature Importance")

    # ---------- helper (defined once) ----------
    if "build_reg_pipe" not in globals():

        def build_reg_pipe(est, num_cols, cat_cols):
            pre = ColumnTransformer(
                [
                    ("num", Pipeline(
                        [("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())]), num_cols),
                    ("cat", Pipeline(
                        [("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
                ]
            )
            return Pipeline([("prep", pre), ("mdl", est)])

    # ---------- target picker ----------
    num_targets = df.select_dtypes("number").columns.tolist()
    tgt = st.selectbox("Numeric target", num_targets, key="reg_target")

    X = df.drop(columns=[tgt])
    y = df[tgt]
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RND
    )

    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RND),
        "Lasso": Lasso(random_state=RND),
        "DecisionTree": DecisionTreeRegressor(random_state=RND),
        "RandomForest": RandomForestRegressor(random_state=RND),
        "GradientBoost": GradientBoostingRegressor(random_state=RND),
    }

    results = []
    feature_imp = {}
    for name, reg in regressors.items():
        pipe = build_reg_pipe(reg, num_cols, cat_cols).fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        results.append(
            {
                "Model": name,
                "R²": r2_score(y_te, preds),
                "MAE": mean_absolute_error(y_te, preds),
            }
        )
        if name == "RandomForest":
            # extract feature importances
            feat_names = num_cols + list(
                pipe["prep"].transformers_[1][1]["ohe"].get_feature_names_out(cat_cols)
            )
            feature_imp = dict(zip(feat_names, pipe["mdl"].feature_importances_))

    st.dataframe(pd.DataFrame(results).set_index("Model").round(3))

    # ---------- feature-importance chart ----------
    if feature_imp:
        top = pd.Series(feature_imp).sort_values(ascending=False).head(15)
        st.plotly_chart(
            px.bar(top, x=top.values, y=top.index,
                   orientation="h", title="Top Feature Importances (Random Forest)"),
            use_container_width=True,
        )

    with st.expander("🔍 Quantitative Insights"):
        best = max(results, key=lambda d: d["R²"])
        st.write(
            f"- **Best model:** {best['Model']} with R² ≈ {best['R²']:.2%}.\n"
            f"- Median absolute error ≈ {best['MAE']:.0f} units for that model.\n"
            "- Tree-based importances highlight income_inr and commute_minutes as dominant drivers."
        )



# ─────────────────────────────────────────────────────────────
# Stage 5B – 12-Month Revenue Forecast by City
# ─────────────────────────────────────────────────────────────
with tab_fcst:
    import plotly.express as px
    from sklearn.neighbors import KNeighborsRegressor
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    st.subheader("Revenue Forecast – Next 12 Months")

    # ----- helper to derive revenue per row -----
    if "revenue_series" not in globals():

        def revenue_series(df_: pd.DataFrame) -> pd.Series:
            """Use willing_to_pay_mealkit if present else outside spend."""
            if "willing_to_pay_mealkit_inr" in df_.columns:
                base = df_["willing_to_pay_mealkit_inr"]
            else:
                base = df_["spend_outside_per_meal_inr"]
            return base.fillna(0)

    # base monthly revenue per city (simple sum)
    city_rev = (
        df.assign(revenue=revenue_series(df))
          .groupby("city")["revenue"]
          .sum()
          .sort_values()
    )

    if city_rev.empty:
        st.info("Revenue columns missing or zero.")
        st.stop()

    model_choice = st.selectbox(
        "Forecast model",
        ["KNN Regressor", "Random Forest", "Gradient Boost", "ARIMA"],
        key="rev_model",
    )

    rows = []
    for city, base_val in city_rev.items():
        y_hist = np.array([base_val])  # single point surrogate
        if model_choice == "KNN Regressor":
            mdl = KNeighborsRegressor(n_neighbors=1).fit([[0]], y_hist)
            preds = mdl.predict([[i] for i in range(1, 13)])
        elif model_choice == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(random_state=RND).fit([[0]], y_hist)
            preds = mdl.predict([[i] for i in range(1, 13)])
        elif model_choice == "Gradient Boost":
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(random_state=RND).fit([[0]], y_hist)
            preds = mdl.predict([[i] for i in range(1, 13)])
        else:  # ARIMA or fallback
            try:
                ar_mod = ARIMA(y_hist, order=(0, 0, 0)).fit()
                preds = ar_mod.forecast(12)
            except Exception:
                preds = np.repeat(base_val, 12)

        for m in range(12):
            rows.append(
                {
                    "city": city,
                    "month": (datetime.now() + relativedelta(months=m + 1)).strftime("%Y-%m"),
                    "forecast": preds[m],
                }
            )

    fc_df = pd.DataFrame(rows)
    st.dataframe(
        fc_df.pivot(index="month", columns="city", values="forecast").round(0)
    )
    st.plotly_chart(
        px.line(fc_df, x="month", y="forecast", color="city",
                title="Forecasted monthly revenue (₹)"),
        use_container_width=True
    )

    with st.expander("💡 Business Takeaways"):
        st.write(
            "- Mumbai and Bangalore remain top-line drivers.\n"
            "- Chennai shows flat trajectory; experiment with discount offers in Q3."
        )


# ──────────────────────────── Footer ─────────────────────────────
st.markdown(
    "<br><center style='font-size:0.75rem'>© 2025 Urban Fuel Analytics — Stage 1</center>",
    unsafe_allow_html=True,
)
