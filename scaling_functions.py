import time
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils import new_line, short_list
from progress_bar import progress_bar
from recipe import record_step

def recommend_scaling(s):
    # Rule of thumb from the column's values: (recommended scaling, reason)
    s = s.dropna()
    if s.nunique() <= 2:
        return "No scaling needed", "only two distinct values (e.g. 0/1)"
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_share = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
    if outlier_share >= 0.05:
        return "Robust Scaling", f"{outlier_share:.0%} of the values are outliers; robust scaling uses the median and IQR, so outliers barely affect it"
    if abs(s.skew()) <= 0.5:
        return "Standard Scaling", "roughly symmetric values with few outliers"
    return "MinMax Scaling", "skewed values with few outliers; MinMax maps them to 0–1 and keeps their shape"

def scaling_recommendations(df, features):
    rows = []
    for feat in features:
        s = df[feat].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
        method, reason = recommend_scaling(s)
        rows.append({"Feature": feat, "Min": round(float(s.min()), 3), "Max": round(float(s.max()), 3),
                     "Outliers": f"{outliers:.1%}", "Skewness": round(float(s.skew()), 2) if s.nunique() > 2 else None,
                     "Recommended": method, "Why": reason})
    return pd.DataFrame(rows)

def overall_scaling_recommendation(df, features):
    # One scaling method that suits all the selected features best (binary features don't need scaling)
    non_binary = [f for f in features if df[f].dropna().nunique() > 2]
    binary = [f for f in features if f not in non_binary]
    if not non_binary:
        return "**No scaling needed**: all selected features have only two values (e.g. 0/1)."
    methods = {f: recommend_scaling(df[f])[0] for f in non_binary}
    with_outliers = [f for f, m in methods.items() if m == "Robust Scaling"]
    if with_outliers:
        text = (f"**Robust Scaling** suits the selected features best overall: {short_list(with_outliers)} {'has' if len(with_outliers) == 1 else 'have'} many outliers, "
                "and robust scaling (median and IQR) is barely affected by them while still working well for the other features.")
    elif all(m == "Standard Scaling" for m in methods.values()):
        text = "**Standard Scaling** suits the selected features best overall: they are all roughly symmetric with few outliers."
    else:
        text = ("**MinMax Scaling** suits the selected features best overall: some are skewed (with few outliers), and MinMax "
                "maps all of them to 0–1 while keeping their shape.")
    if binary:
        text += (f" {short_list(binary)} only has two values and doesn't need scaling; you can leave it out." if len(binary) == 1 else
                 f" {short_list(binary)} only have two values and don't need scaling; you can leave them out.")
    return text

def display_scaling_options(st, df):
        # Scaling Methods
        scaling_methods = st.checkbox("Explain Scaling Methods", value=False, key='scaling_methods')
        if scaling_methods:
            new_line()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h6 align='center'> Standard Scaling </h6>" ,unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - \mu}{\sigma}''')
                new_line()
                # Values Ranges for the output of Standard Scaling in general
                st.latex(r'''z \in [-3,3]''')   

            with col2:
                st.markdown("<h6 align='center'> MinMax Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - min(x)}{max(x) - min(x)}''')
                new_line()
                # Values Ranges for the output of MinMax Scaling in general
                st.latex(r'''z \in [0,1]''')
                
            with col3:
                st.markdown("<h6 align='center'> Robust Scaling </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \frac{x - Q_1}{Q_3 - Q_1}''')
                # Values Ranges for the output of Robust Scaling in general
                new_line()
                st.latex(r'''z \in [-2,2]''')

            # write z in the range for the output in latex
            st.latex(r''' **  Z = The\ Scaled\ Value  ** ''')

            new_line()


        # Ranges for the numeric features
        feat_range = st.checkbox("Further Analysis", value=False, key='feat_range')
        if feat_range:
            new_line()
            st.write("The Ranges for the numeric features:")
            col1, col2, col3 = st.columns([0.05,1, 0.05])
            with col2:
                 st.dataframe(df.describe().T, width=700)
            
            new_line()

        # INPUT
        new_line()
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

        with col2:
            scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling", "Robust Scaling"], help="Select the scaling method.")

        # Recommendation: which scaling suits each selected feature
        if len(scale_feat) == 1:
            recommendations = scaling_recommendations(df, scale_feat)
            if not recommendations.empty:
                st.markdown("💡 **Recommended scaling** for the selected feature:")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
        elif len(scale_feat) > 1:
            st.info(f"💡 **Recommendation:** {overall_scaling_recommendation(df, scale_feat)}")
        if scale_feat:
            st.caption("Tree-based models (LightGBM, XGBoost, Random Forest, Extra Trees, CatBoost) don't need scaling. "
                       "Scaling mainly helps Logistic Regression and K-Nearest Neighbors.")

        # Features with missing values can't be scaled
        missing_cols = [col for col in scale_feat if df[col].isnull().any()]
        if missing_cols:
            st.warning(f"The Features **`{missing_cols}`** have missing values, so they can't be scaled yet. Please handle them in the **⚠️ Missing Values** section first.")

        if scale_feat and scaling != "Select":
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])

                if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling.", disabled=bool(missing_cols)):

                    progress_bar()
    
                    # Standard Scaling
                    if scaling == "Standard Scaling":
                        st.session_state.all_the_process += f"""
# Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns, index=df.index)
\n """
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns, index=df.index)
                        record_step({"op": "scale", "columns": list(scale_feat), "scaler": scaler, "label": f"{scaling} of {list(scale_feat)}"})
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
    
                    # MinMax Scaling
                    elif scaling == "MinMax Scaling":
                        st.session_state.all_the_process += f"""
# MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns, index=df.index)
\n """
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns, index=df.index)
                        record_step({"op": "scale", "columns": list(scale_feat), "scaler": scaler, "label": f"{scaling} of {list(scale_feat)}"})
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
    
                    # Robust Scaling
                    elif scaling == "Robust Scaling":
                        st.session_state.all_the_process += f"""
# Robust Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns, index=df.index)
\n """
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns, index=df.index)
                        record_step({"op": "scale", "columns": list(scale_feat), "scaler": scaler, "label": f"{scaling} of {list(scale_feat)}"})
                        st.session_state['df'] = df
                        st.success(f"The Features **`{scale_feat}`** have been scaled using Robust Scaling.")

        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
        if show_df:
            st.dataframe(df, use_container_width=True)
