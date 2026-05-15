# 📊 Click Analytics App

**A web app for exploratory data analytics**

## 🚀 Features

### 📁 Data Handling
- Upload CSV datasets
- Preview data and schema
- Handle missing values (mean, median, mode, etc.)

### 🔍 Exploratory Data Analysis (EDA)
- Summary statistics
- Distribution plots
- Correlation analysis
- Missing value visualizations
- Interactive Plotly charts

### 🧹 Feature Engineering
- Encoding categorical features
- Scaling and transformations
- Safe log / power transformations

### ✂️ Data Splitting
- Train / Test split
- Train / Validation / Test split
- Download split datasets as CSV

### 🤖 Model Building
- Train ML models using:
  - XGBoost
  - LightGBM
  - CatBoost
  - Scikit-learn models
- Automatic handling of categorical features
- Session-based model tracking

### 📈 Model Evaluation
- Classification & regression metrics
- ROC Curve visualization
- Predictions on train / validation / test sets

---

## 🛠️ Tech Stack

- **Frontend / App:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:**  
  - Scikit-learn  
  - XGBoost  
  - LightGBM  
  - CatBoost
- **Deployment:** Streamlit Cloud
- **Python Version:** 3.11

---

## 📦 Installation (Local)

```bash
git clone https://github.com/AUNGNYILATT421/Click_Analytics_App.git
cd Click_Analytics_App
pip install -r requirements.txt
streamlit run app.py
