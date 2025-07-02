# README.md

## Urban Fuel – Consumer Intelligence Hub 🍱

<img src="assets/preview.png" width="100%">

### Overview
An all-in-one BI portal for the **Urban Fuel synthetic survey**.  
Features six richly-interactive tabs:

| Tab | Highlights |
|-----|------------|
| **Exploration** | ≥20 premium visuals, executive captions, dark-mode ready |
| **Classification** | 5–6 algorithms, ROC overlay, confusion matrix, batch prediction |
| **Clustering** | K-means + elbow & silhouette plots, persona explorer |
| **Association Rules** | Apriori with user thresholds, rule table & network graph |
| **Regression** | 6 regressors, KPI table, feature-importance bar |
| **Forecast** | 12-month city revenue using KNN/RF/GB/ARIMA/Prophet |

### Quick-start (local)

```bash
git clone https://github.com/<you>/urban-fuel-dash.git
cd urban-fuel-dash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
