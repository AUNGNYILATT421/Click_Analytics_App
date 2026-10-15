# 📊 Click Analytics App

**A no-code web app for data analysis and automated machine learning, built with Streamlit.**

Upload a dataset, explore and clean it, prepare the features, and let automated machine learning find the best model — all in the browser, without writing code.

🔗 **Live app:** [clickanalyticsapp-ezio.streamlit.app](https://clickanalyticsapp-ezio.streamlit.app)

---

## 🚀 Features

### 📁 Data Loading

- Upload **CSV, XLSX, or XLS** files, or load them from a **URL**
- **Google Drive** share links are supported (the file must be shared as "Anyone with the link")

### 🔍 Exploratory Data Analysis (EDA)

- First/last rows, shape, columns, data types, summary statistics, skewness and kurtosis, unique value counts
- Correlation heatmap and missing-value charts
- Histograms, box plots, scatter plots, pair plots, count plots, pie charts, outlier plots
- Cross-tabulations, segmented analysis, temporal analysis (datetime columns), word cloud

### 🧽 Data Cleaning

- Delete columns, or delete rows by row number (e.g. `3, 7, 10-15`)
- Show and remove duplicate rows
- Change data types (integer, float, text, boolean, datetime) — conversions that would lose data are blocked with an explanation

### ⚠️ Missing Values

- Treat blank text (empty or only spaces) as missing
- Find the rows with missing values in a column
- Drop rows or columns, or fill with mean, median, mode, or forward/backward fill

### 🧹 Feature Preparation

- **Encoding:** ordinal, one-hot (first category dropped), count frequency
- **Scaling:** standard, MinMax, robust
- **Transformations:** log, square root, cube root, exponential
- **Feature engineering:** create features from two columns (+, −, ×, ÷), transform a feature by a constant, select features

### ✂️ Data Splitting

- Train / Test or Train / Validation / Test split, with each set downloadable as CSV
- Features that are 100% correlated with the target are removed automatically (they would give the answer away)
- If the data changes after splitting, the old split and any trained model are discarded, so a model is never trained on outdated data

### 🤖 Automated Model Training ([FLAML](https://github.com/microsoft/FLAML))

- Choose the problem type (classification or regression), an **optimization metric**, a **time budget** (10–300 s), and the **model types to try**: LightGBM, XGBoost, Random Forest, Extra Trees, CatBoost, Logistic Regression, K-Nearest Neighbors
- FLAML tries the model types, tunes their settings within the time budget, and keeps the best model
- A **leaderboard** shows the best validation score of each model type
- Optional missing-value filling and scaling, learned from the training set only (no test data leaks into training)
- Checks before training explain in plain language what must be fixed first: text columns, missing values, duplicate rows (users may choose to keep them), or duplicate column names

### 📈 Model Evaluation

- **Classification:** accuracy, precision, recall, F1, AUC; ROC curve (one curve per class for more than two classes); confusion matrix
- **Regression:** MAE, MSE, RMSE, R²
- **Permutation feature importance** chart
- Download the trained model (`.pkl`) and the processed dataset (`.csv`)

---

## 🛠️ Tech Stack

- **App:** Streamlit
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, WordCloud
- **Machine learning:** scikit-learn, FLAML, LightGBM, XGBoost, CatBoost
- **Deployment:** Streamlit Community Cloud
- **Python:** 3.12

All package versions are pinned in [`requirements.txt`](requirements.txt) to a combination tested together.

---

## 📦 Installation (Local)

Requires **Python 3.12**.

```bash
git clone https://github.com/AUNGNYILATT421/Click_Analytics_App.git
cd Click_Analytics_App
python3.12 -m venv my_env
source my_env/bin/activate        # Windows: my_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Using a Downloaded Model

The downloaded `model.pkl` is a FLAML model (wrapped in a scikit-learn `Pipeline` if missing-value filling or scaling was used). Load it in Python with the same package versions as in `requirements.txt`:

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")

new_data = pd.read_csv("new_data.csv")   # same columns, prepared the same way as in the app
predictions = model.predict(new_data)
```

New data must go through the same preparation steps used in the app (cleaning, encoding, transformations) before prediction.

---

## 📄 License

[MIT](LICENSE)
