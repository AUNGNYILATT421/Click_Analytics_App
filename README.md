# 📊 Click Analytics App

**A no-code web app for data analysis and automated machine learning, built with Streamlit.**

Upload a dataset, explore and clean it, prepare the features, and let automated machine learning find the best model — all in the browser, without writing code.

---

## 🚀 Features

### 🧭 Workflow Guide

- A checklist in the sidebar shows the usual order of the steps (load → explore → missing values → prepare → split → train → evaluate → predict), ticks the ones that are done, and suggests where to go next
- A guide only: every section stays on the page and can be used at any time

### 📁 Data Loading

- Upload **CSV, XLSX, or XLS** files, or load them from a **URL**
- **Google Drive** share links are supported (the file must be shared as "Anyone with the link")
- Try the app with built-in **sample datasets** (Breast Cancer, Wine, Diabetes), which need no download

### 🩺 Data Check

- Right after loading, plain-language warnings about common causes of poor or misleading models: identifier columns (IDs, names, row numbers), constant columns, text with too many different values, mostly empty columns, and duplicate rows
- Suggestions only: you decide what to change (large data is checked on a 100,000-row sample)

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
- **Steps Applied** (sidebar): every change to the data is listed in order, and **Undo Last Step** restores the data as it was before it (up to 5 steps in a row; copies are capped at 200 MB in total, so undo is unavailable for very large data). Undoing after a split discards the split and any trained model
- **Recommendations** based on the selected columns' values (for one column, or one overall method for several): Ordinal instead of One Hot Encoding when it would create many columns, which scaling method suits the data, and which transformations to avoid (e.g. log for values ≤ 0)

### ✂️ Data Splitting

- Train / Test or Train / Validation / Test split, with each set downloadable as CSV
- Features that are 100% correlated with the target are removed automatically (they would give the answer away); features at least 95% correlated get a warning, since they are often built from the target or recorded after it
- If the data changes after splitting, the old split and any trained model are discarded, so a model is never trained on outdated data

### 🤖 Automated Model Training ([FLAML](https://github.com/microsoft/FLAML))

- Choose the problem type (classification or regression), an **optimization metric**, a **time budget** (10–300 s), and the **model types to try**: LightGBM, XGBoost, Random Forest, Extra Trees, CatBoost, Logistic Regression, K-Nearest Neighbors
- FLAML tries the model types, tunes their settings within the time budget, and keeps the best model
- A **leaderboard** shows the best validation score of each model type
- **Imbalanced classes** (a class under 20% of the training rows) are flagged, with a suggestion to optimise F1 or ROC AUC and an optional **Balance classes** setting (balanced sample weights; no rows are added or removed)
- Optional missing-value filling and scaling, learned from the training set only (no test data leaks into training)
- Checks before training explain in plain language what must be fixed first: text columns, missing values, duplicate rows (users may choose to keep them), or duplicate column names

### 📈 Model Evaluation

- **Classification:** accuracy, precision, recall, F1, AUC; ROC curve (one curve per class for more than two classes); confusion matrix
- **Regression:** MAE, MSE, RMSE, R²
- **Permutation feature importance** chart
- Download the trained model (`.pkl`, bundled with the recorded preparation steps) and the processed dataset (`.csv`)

### 🔮 Prediction

- Predict the target for **one new row**: enter a value for each original column the model needs (numbers, or a dropdown of the known categories)
- The row is prepared with **exactly the same steps as the training data**, using the values learned from it (e.g. the same fill values, category mappings, and scaling) — nothing is re-learned from the new row
- Shows the prediction (with class probabilities for classification) and the list of steps applied
- Prediction is disabled, with an explanation, when a field is empty, when a category was never seen in training, or when a feature is built from the target itself
- Predict **many rows from a file** (CSV or Excel, up to 100,000 rows): every row is prepared the same way, and the downloadable result has Prediction (and probability) columns first. Rows that can't be predicted are not dropped: a **Problem** column says why (e.g. an empty field, text in a number column, or a category never seen in training)

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

The downloaded `model.pkl` contains the trained model **and the recorded preparation steps**, so raw new data (with the original columns, as in the file you loaded) can be predicted outside the app. From this app's folder, with the packages in `requirements.txt` installed:

```bash
python predict_with_bundle.py model.pkl new_data.csv predictions.csv
```

The output has the same rows as `new_data.csv` with Prediction (and probability) columns and a Problem column, exactly like the file prediction in the app.

Only load `.pkl` files you created yourself: loading a pickle file can run code stored in it. For the same reason the app doesn't accept uploaded model files.

---

## 📄 License

[MIT](LICENSE)
