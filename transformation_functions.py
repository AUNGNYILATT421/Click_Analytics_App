import streamlit as st
import numpy as np
from utils import new_line, short_list
import time
from progress_bar import progress_bar

# Exponential Transformation is not recommended when values are larger than this in magnitude
EXP_LARGE_VALUE = 10

def unsuitable_for(values, trans):
    # Whether a transformation is not recommended for these (non-missing) values
    if trans == "Log Transformation":
        return values.min() <= 0
    if trans == "Square Root Transformation":
        return values.min() < 0
    if trans == "Exponential Transformation":
        return values.abs().max() > EXP_LARGE_VALUE
    return False

def overall_transformation_recommendation(df, features, trans):
    # If the chosen transformation doesn't suit every feature, recommend the strongest one that suits them all
    columns = {f: df[f].dropna() for f in features if not df[f].dropna().empty}
    unsuitable = [f for f, v in columns.items() if unsuitable_for(v, trans)]
    if not unsuitable:
        return None
    reason = ("it has " if len(unsuitable) == 1 else "they have ") + {
        "Log Transformation": "values ≤ 0",
        "Square Root Transformation": "negative values",
        "Exponential Transformation": "numerically large values (e^x explodes above about 709 and shrinks large negative values to almost 0)"}[trans]
    message = f"**{trans}** isn't suitable for **{len(unsuitable)} of {len(columns)}** selected features ({short_list(unsuitable)}): {reason}. "
    if trans == "Exponential Transformation":
        # Exponential expands values; no other transformation does the same job, so only advise avoiding it
        return message + "It is not recommended for these features."
    smallest = min(v.min() for v in columns.values())
    best = "Log Transformation" if smallest > 0 else "Square Root Transformation" if smallest >= 0 else "Cube Root Transformation"
    return message + f"**{best}** is recommended overall, as it suits all the selected features."

def display_transformation_options(st, df):
    # Data Transformation
    new_line()
    st.markdown("### 🧬 Data Transformation", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Data Transformation"):
        new_line()

        # Transformation Methods
        trans_methods = st.checkbox("Explain Transformation Methods", key="trans_methods", value=False)
        if trans_methods:
            new_line()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("<h6 align='center'> Log <br> Transformation</h6>", unsafe_allow_html=True)
                st.latex(r'''z = log(x)''')

            with col2:
                st.markdown("<h6 align='center'> Square Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt{x}''')

            with col3:
                st.markdown("<h6 align='center'> Cube Root Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = \sqrt[3]{x}''')

            with col4:
                st.markdown("<h6 align='center'> Exponential Transformation </h6>", unsafe_allow_html=True)
                st.latex(r'''z = e^x''')

        # Input
        new_line()
        col1, col2 = st.columns(2)
        with col1:
            trans_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features you want to transform.", key="transformation features")

        with col2:
            trans = st.selectbox("Select Transformation", ["Select", "Log Transformation", "Square Root Transformation", "Cube Root Transformation", "Exponential Transformation"],
                                 help="Select the transformation you want to apply.", 
                                 key="transformation")

        # Features with missing values can't be transformed
        missing_cols = [col for col in trans_feat if df[col].isnull().any()]
        if missing_cols:
            st.warning(f"The Features **`{missing_cols}`** have missing values, so they can't be transformed yet. Please handle them in the **⚠️ Missing Values** section first.")

        # The log isn't defined for negative numbers
        negative_cols = []
        if trans == "Log Transformation":
            negative_cols = [col for col in trans_feat if (df[col] < 0).any()]
            if negative_cols:
                st.warning(f"The Features **`{negative_cols}`** have negative values, and the log isn't defined for negative numbers. Please choose another transformation, such as **Cube Root Transformation**, or select other features.")

        # Recommendations: transformations to avoid for these values (advice only; negative values for Log are blocked above)
        advice = []
        if len(trans_feat) > 1 and trans != "Select":
            # One overall recommendation for several features
            overall = overall_transformation_recommendation(df, trans_feat, trans)
            if overall:
                st.info(f"💡 **Recommendation:** {overall}")
        for col in (trans_feat if len(trans_feat) == 1 else []):
            values = df[col].dropna()
            if values.empty:
                continue
            if trans == "Log Transformation" and col not in negative_cols and values.min() <= 0:
                advice.append(f"**`{col}`** has values ≤ 0 (smallest: {values.min():g}). Log Transformation is best for positive values; "
                              "consider **Cube Root Transformation**.")
            elif trans == "Square Root Transformation" and values.min() < 0:
                advice.append(f"**`{col}`** has negative values (smallest: {values.min():g}). Square Root Transformation is meant for "
                              "values ≥ 0; consider **Cube Root Transformation**, which handles negative values naturally.")
            elif trans == "Exponential Transformation" and values.abs().max() > EXP_LARGE_VALUE:
                advice.append(f"**`{col}`** has numerically large values (largest magnitude: {values.abs().max():g}). e^x makes large "
                              "positive values explode (above about 709 they become infinite) and large negative values shrink to "
                              "almost 0, so Exponential Transformation is not recommended here.")
        if advice:
            st.info("💡 **Recommendation:**\n\n" + "\n".join(f"- {a}" for a in advice))

        # Apply Transformation
        if trans_feat and trans != "Select":
            new_line()
            col1, col2, col3 = st.columns([1, 0.5, 1])
            if col2.button("Apply", key='trans_apply',use_container_width=True ,help="Click to apply transformation.", disabled=bool(missing_cols or negative_cols)):

                progress_bar()

                # Depending on the transformation selected
                if trans == "Log Transformation":
                    df[trans_feat] = np.log1p(df[trans_feat].astype(float))
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Log Transformation.")
                elif trans == "Square Root Transformation":
                    x2 = df[trans_feat].astype(float)
                    df[trans_feat] = np.sign(x2) * np.sqrt(np.abs(x2))
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Square Root Transformation.")
                elif trans == "Cube Root Transformation":
                    df[trans_feat] = np.cbrt(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Cube Root Transformation.")
                elif trans == "Exponential Transformation":
                    df[trans_feat] = np.exp(df[trans_feat])
                    st.session_state['df'] = df
                    st.success("Numerical features have been transformed using Exponential Transformation.")

        # Show DataFrame Button
        new_line()
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
        
        if show_df:
            st.dataframe(df, use_container_width=True)

