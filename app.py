# ────────────────────────────────────────────────────────────────────────────────
# app.py  ·  Urban Fuel – Consumer Intelligence Hub  ·  All-Plotly, Full Edition
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "Urban Fuel – Consumer Intelligence Hub"
PAGE_ICON = "🍱"
DATA_PATH = pathlib.Path("UrbanFuelSyntheticSurvey (1).csv")

THEME_PALETTE = px.colors.qualitative.Safe
FORECAST_HORIZON = 12  # months
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    numeric_like = [
        c
        for c in df.select_dtypes(include="object").columns
        if df[c].str.replace(".", "", 1).str.isnumeric().all()
    ]
    df[numeric_like] = df[numeric_like].apply(pd.to_numeric, errors="coerce")
    return df


def format_inr(x) -> str:
    return "-" if pd.isna(x) else f"₹{int(x):,}"


def cache_model(func):
    @st.cache_resource(show_spinner="Training model…")
    def _cached(*args, **kwargs):
        return func(*args, **kwargs)

    return _cached


def split_xy(
    _df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = _df[target]
    X = _df.drop(columns=[target])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num, cat


def build_pipe(est, num_cols, cat_cols) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return Pipeline([("prep", prep), ("mdl", est)])


def metrics_df(y_t, y_p, y_prob) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Accuracy": [metrics.accuracy_score(y_t, y_p)],
            "Precision": [metrics.precision_score(y_t, y_p, zero_division=0)],
            "Recall": [metrics.recall_score(y_t, y_p, zero_division=0)],
            "F1": [metrics.f1_score(y_t, y_p, zero_division=0)],
            "AUC": [
                metrics.roc_auc_score(y_t, y_prob[:, 1]) if y_prob is not None else np.nan
            ],
        }
    ).round(3)


def revenue_proxy(_df: pd.DataFrame) -> pd.Series:
    if "willing_to_pay_mealkit_inr" in _df.columns:
        return _df["willing_to_pay_mealkit_inr"].fillna(0)
    return _df["spend_outside_per_meal_inr"].fillna(0)


def safe_toast(msg: str, icon: str = "⚠️"):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.warning(msg)


# ────────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

df_raw = load_data()

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/simpleicons/simple-icons/develop/icons/bento.svg",
        width=48,
    )
    st.title("Filters")
    city_sel = st.multiselect("City", sorted(df_raw["city"].dropna().unique()))
    gender_sel = st.multiselect("Gender", sorted(df_raw["gender"].dropna().unique()))
    inc_min, inc_max = map(int, [df_raw["income_inr"].min(), df_raw["income_inr"].max()])
    inc_rng = st.slider(
        "Income range (INR)", inc_min, inc_max, (inc_min, inc_max), step=10000
    )
    diet_sel = st.multiselect(
        "Dietary goals", sorted(df_raw["dietary_goals"].dropna().unique())
    )

df = df_raw.copy()
if city_sel:
    df = df[df["city"].isin(city_sel)]
if gender_sel:
    df = df[df["gender"].isin(gender_sel)]
df = df[df["income_inr"].between(inc_rng[0], inc_rng[1])]
if diet_sel:
    df = df[df["dietary_goals"].isin(diet_sel)]

if df.empty:
    safe_toast("No records match selected filters.", "❗")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────────
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
tab_viz, tab_clf, tab_cluster, tab_rules, tab_reg, tab_fcst = tabs

# ────────────────────────────────────────────────────────────────────────────────
# 1 ▸ Data Visualisation
# ────────────────────────────────────────────────────────────────────────────────
with tab_viz:
    st.header("Interactive Insights")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", f"{len(df):,}")
    k2.metric("Avg. Age", f"{df['age'].mean():.1f}")
    k3.metric("Median Income", format_inr(df["income_inr"].median()))
    k4.metric(
        "Healthy-Eating Importance",
        f"{df['healthy_importance_rating'].mean():.2f}/5",
    )

    viz = [
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
            color="gender",
            title="Income by Gender",
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
        px.pie(df, names="meal_type_pref", title="Meal Type Preference"),
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
            color_discrete_sequence=THEME_PALETTE,
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

    for i in range(0, len(viz), 3):
        cols = st.columns(3)
        for v, col in zip(viz[i : i + 3], cols):
            with col:
                st.plotly_chart(v, use_container_width=True)
                st.caption(v.layout.title.text)

    with st.expander("📌 Business Takeaways"):
        st.markdown(
            textwrap.dedent(
                """
                • **Metro cities skew richer**, enabling premium pricing.  
                • Longer commutes ↔ fewer cooked dinners → convenience value-prop.  
                • Health focus increases with age – segment messaging accordingly.  
                • High non-veg clusters suggest protein-rich kit opportunities.
                """
            )
        )

# ────────────────────────────────────────────────────────────────────────────────
# 2 ▸ Classification
# ────────────────────────────────────────────────────────────────────────────────
with tab_clf:
    st.header("Customer Conversion Classification")

    tgt = st.selectbox(
        "Choose binary target label",
        ["subscribe_try", "continue_service", "refer_service"],
    )

    X, y, num_cols, cat_cols = split_xy(df, tgt)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    strat = y if y.nunique() == 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )

    models = {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: Dict[str, Dict] = {}
    roc_fig = go.Figure(layout=go.Layout(title="ROC Curve"))

    for name, est in models.items():

        @cache_model
        def _train(e):
            pipe = build_pipe(e, num_cols, cat_cols)
            pipe.fit(X_tr, y_tr)
            return pipe

        mdl = _train(est)
        y_pred = mdl.predict(X_te)
        try:
            y_prob = mdl.predict_proba(X_te)
        except AttributeError:
            y_prob = None

        results[name] = {
            "model": mdl,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics_df(y_te, y_pred, y_prob),
        }

        if y_prob is not None:
            fpr, tpr, _ = metrics.roc_curve(y_te, y_prob[:, 1])
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))

    st.subheader("Performance Metrics")
    st.dataframe(
        pd.concat(
            [v["metrics"].assign(Model=k) for k, v in results.items()]
        ).set_index("Model")
    )

    st.plotly_chart(
        roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR"),
        use_container_width=True,
    )

    sel = st.selectbox("Confusion matrix for:", list(results.keys()))
    fig_cm = ConfusionMatrixDisplay.from_predictions(
        y_te, results[sel]["y_pred"]
    ).figure_
    st.pyplot(fig_cm)

    st.markdown("---")
    st.subheader("Batch Prediction")
    upl = st.file_uploader("Upload CSV (no target column)", type="csv")
    if upl:
        new = pd.read_csv(upl)
        try:
            preds = results[sel]["model"].predict(new)
            new[f"pred_{tgt}"] = preds
            buff = BytesIO()
            new.to_csv(buff, index=False)
            st.download_button("Download predictions", buff.getvalue(), "preds.csv")
            st.success("✅ Predictions complete")
        except Exception as e:
            safe_toast(f"Prediction failed: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# 3 ▸ Clustering
# ────────────────────────────────────────────────────────────────────────────────
with tab_cluster:
    st.header("Persona Discovery – K-means")

    num_cols_all = df.select_dtypes(include="number").columns.tolist()
    cl_cols = st.multiselect(
        "Numeric features for clustering (2-5)",
        num_cols_all,
        default=num_cols_all[:4],
    )
    if len(cl_cols) < 2:
        st.warning("Pick at least two features.")
        st.stop()

    k_val = st.slider("k (clusters)", 2, 10, 4)
    inertia = [
        KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(df[cl_cols]).inertia_
        for k in range(2, 11)
    ]
    st.plotly_chart(
        px.line(x=list(range(2, 11)), y=inertia, markers=True, title="Elbow Curve"),
        use_container_width=True,
    )

    km = KMeans(n_clusters=k_val, random_state=RANDOM_STATE).fit(df[cl_cols])
    df["cluster"] = km.labels_

    st.plotly_chart(
        px.scatter(
            df,
            x=cl_cols[0],
            y=cl_cols[1],
            color="cluster",
            hover_data=cl_cols,
            title="Cluster Scatter",
            color_continuous_scale="Viridis",
        ),
        use_container_width=True,
    )

    st.subheader("Cluster Personas (mean values)")
    st.dataframe(df.groupby("cluster")[cl_cols].mean().round(1))

    csv_buf = BytesIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download clustered data", csv_buf.getvalue(), "clustered.csv")

# ────────────────────────────────────────────────────────────────────────────────
# 4 ▸ Association Rules
# ────────────────────────────────────────────────────────────────────────────────
with tab_rules:
    st.header("Market-Basket Insights (Apriori)")

    cat_opts = df.select_dtypes(exclude="number").columns.tolist()
    fields = st.multiselect(
        "Pick up to three categorical fields", cat_opts, default=cat_opts[:3]
    )
    sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if not fields:
        st.info("Select categorical columns.")
    else:
        trans = (
            df[fields]
            .astype(str)
            .apply(lambda c: "___" + c.name + "__" + c.str.strip())
        )
        basket = pd.get_dummies(trans.stack()).groupby(level=0).sum().astype(bool)
        freq = apriori(basket, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"] >= lift]
        if rules.empty:
            st.info("No rules meet the thresholds.")
        else:
            st.dataframe(
                rules.sort_values("confidence", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

# ────────────────────────────────────────────────────────────────────────────────
# 5 ▸ Regression & Feature Importance
# ────────────────────────────────────────────────────────────────────────────────
with tab_reg:
    st.header("Drivers of Continuous Outcomes")

    numeric_targets = [
        c for c in df.select_dtypes(include="number").columns if c != "cluster"
    ]
    tgt_reg = st.selectbox("Select numeric target", numeric_targets)

    Xr, yr, num_r, cat_r = split_xy(df, tgt_reg)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    }

    res, fi = [], {}
    for name, reg in regs.items():

        @cache_model
        def _train_r(e):
            return build_pipe(e, num_r, cat_r)

        rp = _train_r(reg)
        rp.fit(Xr, yr)
        preds = rp.predict(Xr)
        res.append(
            {"Model": name, "R²": round(metrics.r2_score(yr, preds), 3), "MAE": round(metrics.mean_absolute_error(yr, preds), 1)}
        )

        if name == "Decision Tree":
            imp = rp["mdl"].feature_importances_
            ohe_names = (
                rp["prep"]
                .transformers_[1][1]["ohe"]
                .get_feature_names_out(cat_r)
                .tolist()
            )
            fi = dict(zip(num_r + ohe_names, imp))

    st.dataframe(pd.DataFrame(res).set_index("Model"))

    if fi:
        imp_ser = pd.Series(fi).sort_values(ascending=False).head(15)
        st.plotly_chart(
            px.bar(
                imp_ser,
                x=imp_ser.values,
                y=imp_ser.index,
                orientation="h",
                title="Top Feature Importances (Decision Tree)",
            ),
            use_container_width=True,
        )

# ────────────────────────────────────────────────────────────────────────────────
# 6 ▸ 12-Month Revenue Forecast
# ────────────────────────────────────────────────────────────────────────────────
with tab_fcst:
    st.header("Revenue Forecast by City (12 Months)")

    rev_col = revenue_proxy(df)
    city_rev = df.assign(revenue=rev_col).groupby("city")["revenue"].sum()

    mdl_choice = st.selectbox("Model", ["Random Forest", "ARIMA"])
    future = []

    for city, base_val in city_rev.items():
        if mdl_choice == "Random Forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE).fit(
                np.arange(len(city_rev)).reshape(-1, 1), city_rev.values
            )
            preds = rf.predict(
                np.arange(len(city_rev), len(city_rev) + FORECAST_HORIZON).reshape(-1, 1)
            )
        else:
            ar = ARIMA(np.array(city_rev), order=(1, 1, 0)).fit()
            preds = ar.forecast(FORECAST_HORIZON)

        for i in range(FORECAST_HORIZON):
            month = (datetime.now() + relativedelta(months=i + 1)).strftime("%Y-%m")
            future.append({"city": city, "month": month, "forecast": preds[i]})

    fdf = pd.DataFrame(future)
    st.dataframe(
        fdf.pivot(index="month", columns="city", values="forecast").round(0).style.format(format_inr)
    )

    st.plotly_chart(
        px.line(fdf, x="month", y="forecast", color="city", title="Forecasted Revenue"),
        use_container_width=True,
    )

# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center>© 2025 Urban Fuel Analytics · Built with Streamlit</center>",
    unsafe_allow_html=True,
)
