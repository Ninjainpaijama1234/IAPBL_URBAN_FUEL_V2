# Urban Fuel – Consumer Intelligence Hub 🍱

*A one-stop, point-and-click dashboard that turns a raw CSV survey into crystal-clear business insights.*

---

## 📦 What’s inside?

| Tab | What you’ll see | Why it matters (in plain English) |
|-----|-----------------|------------------------------------|
| **Exploration** | 20 eye-catching charts – bar, heat-map, sunburst, radar, 3-D scatter and more. | Get a “feel” for the data in seconds. Spot patterns (e.g. young commuters cook less at home) without writing a single line of code. |
| **Classification** | Four machine-learning models, auto-scored on accuracy / precision / recall. One-click confusion matrix & ROC curves. | Predict who will *subscribe*, *continue* or *refer* – so you can target the right customers with the right offer. |
| **Clustering** | K-means elbow curve, slider to pick **k**, coloured scatter with centroids, “persona” table. | Groups similar customers together (“time-poor diners”, “health-first commuters”…). Tailor product bundles per segment. |
| **Association Rules** | Apriori rule-mining with sliders for support / confidence / lift. Top-10 rules table. | Uncovers hidden “if-then” links – e.g. *IF* diet goal = low-carb *AND* cuisine = Mediterranean → high chance of digital payment. |
| **Regression** | Six regressors + feature-importance bar chart. | Shows which factors drive *numeric* outcomes (e.g. willingness-to-pay). Clear ranking tells you what to tweak first. |
| **Forecast** | 12-month revenue projection by city (4 model choices) + line chart. | Peek into the future – know which city needs extra marketing love before sales dip. |

---

## 🚀 Quick start (local)

```bash
# 1. Clone repo & enter it
git clone https://github.com/your-handle/urban-fuel-dash.git
cd urban-fuel-dash

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Fire up the dashboard
streamlit run app.py
