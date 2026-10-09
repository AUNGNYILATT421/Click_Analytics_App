import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from utils import new_line
from progress_bar import progress_bar

def fill_mean_median(df, features, stat):
    # Fill numeric features with the mean/median and non-numeric ones with the mode, in place
    features = list(dict.fromkeys(features))  # drop duplicates, keep order
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]

    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].agg(stat))
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    return num_cols, cat_cols

def blank_mask(s):
    # True for text values that are empty or contain only spaces
    if not (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
        return pd.Series(False, index=s.index)
    return s.apply(lambda v: isinstance(v, str) and v.strip() == "")

def format_row_ranges(rows):
    # [5, 17, 18, 19, 42] -> "5, 17-19, 42" (the format Delete Rows accepts)
    parts, start = [], None
    for i, r in enumerate(rows):
        if start is None:
            start = r
        if i == len(rows) - 1 or rows[i + 1] != r + 1:
            parts.append(str(start) if start == r else f"{start}-{r}")
            start = None
    return ", ".join(parts)

def handle_missing_values(df):
    # Missing Values
    new_line()
    st.markdown("### ⚠️ Missing Values", unsafe_allow_html=True)
    new_line()
    with st.expander("Show Missing Values"):

        # Treat Blank Text as Missing (empty strings and strings with only spaces)
        new_line()
        treat_blank = st.checkbox("Treat Blank Text as Missing", value=False, key='treat_blank')
        if treat_blank:
            blank_counts = {col: int(blank_mask(df[col]).sum()) for col in df.columns}
            blank_counts = {col: n for col, n in blank_counts.items() if n}
            if not blank_counts:
                st.info("There are no blank text values (empty or only spaces).")
            else:
                st.write("Columns with blank text values (empty or only spaces): "
                         + ", ".join(f"**`{col}`** ({n})" for col, n in blank_counts.items()))
                blank_cols = st.multiselect("Select Columns", list(blank_counts), key='blank_cols',
                                            help="Blank text in these columns will become missing values.")
                if blank_cols:
                    col1, col2, col3 = st.columns([1, 0.7, 1])
                    if col2.button("Convert to Missing", use_container_width=True, key='blank_apply'):
                        st.session_state.all_the_process += f"""
# Treat Blank Text as Missing
for col in {blank_cols}:
    df.loc[df[col].apply(lambda v: isinstance(v, str) and v.strip() == ""), col] = np.nan
\n """
                        progress_bar()
                        converted = 0
                        for col in blank_cols:
                            mask = blank_mask(df[col])
                            converted += int(mask.sum())
                            df.loc[mask, col] = np.nan
                        st.session_state['df'] = df
                        st.success(f"**{converted}** blank text values in **`{blank_cols}`** are now missing values.")

        # Further Analysis
        new_line()
        missing = st.checkbox("Further Analysis", value=False, key='missing')
        new_line()
        if missing:

            col1, col2 = st.columns(2, gap='medium')
            with col1:
                # Number of Null Values
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False), height=300, use_container_width=True)

            with col2:
                # Percentage of Null Values
                st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                null_percentage = pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100, 2))
                null_percentage.columns = ['Percentage']
                # Sort while the values are still numbers, then format them as text
                null_percentage = null_percentage.sort_values(by='Percentage', ascending=False)
                null_percentage['Percentage'] = null_percentage['Percentage'].map('{:.2f} %'.format)
                st.dataframe(null_percentage, height=300, use_container_width=True)

            # Heatmap
            col1, col2, col3 = st.columns([0.1,1,0.1])
            with col2:
                new_line()
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True, key='null_values_heatmap')

        # Find Missing Rows: the row numbers with missing values in one column
        find_missing = st.checkbox("Find Missing Rows", value=False, key='find_missing')
        new_line()
        if find_missing:
            col_missing = st.selectbox("Select Column", df.columns, key='find_missing_col')
            rows = df.index[df[col_missing].isnull()].tolist()

            n_blank = int(blank_mask(df[col_missing]).sum())
            if n_blank:
                st.info(f"**`{col_missing}`** also has **{n_blank}** blank text values that aren't counted as missing yet. "
                        "Use **Treat Blank Text as Missing** above to include them.")

            if not rows:
                st.info(f"**`{col_missing}`** has no missing values.")
            else:
                st.write(f"**{len(rows)}** rows have missing values in **`{col_missing}`**. "
                         "Row numbers (you can paste them into **🕵️‍♂️ Exploratory Data Analysis → Delete Rows**):")
                st.code(format_row_ranges(rows), language=None)
                st.dataframe(df.loc[rows], use_container_width=True, height=min(35 * len(rows) + 38, 300))
            new_line()


        # INPUT
        col1, col2 = st.columns(2)
        with col1:
            missing_df_cols = df.columns[df.isnull().any()].tolist()
            if missing_df_cols:
                add_opt = ["All Numerical Features (Default Feature)", "All Categorical Feature (Default Feature)"]
            else:
                add_opt = []
            fill_feat = st.multiselect("Select Features",  missing_df_cols + add_opt ,  help="Select Features to fill missing values")

        with col2:
            strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)", "Fill with ffill, bfill"], help="Select Missing Values Strategy")


        if fill_feat and strategy != "Select":

            new_line()
            col1, col2, col3 = st.columns([1,0.5,1])
            if col2.button("Apply", use_container_width=True, key="missing_apply", help="Apply Missing Values Strategy"):

                progress_bar()
                
                # All Numerical Features
                if "All Numerical Features (Default Feature)" in fill_feat:
                    fill_feat.remove("All Numerical Features (Default Feature)")
                    fill_feat += df.select_dtypes(include=np.number).columns.tolist()

                # All Categorical Features
                if "All Categorical Feature (Default Feature)" in fill_feat:
                    fill_feat.remove("All Categorical Feature (Default Feature)")
                    fill_feat += df.select_dtypes(include=object).columns.tolist()

                
                # Drop Rows
                # Changes are made in place so the rest of the page sees them in this run
                fill_feat = list(dict.fromkeys(fill_feat))
                if strategy == "Drop Rows":
                    st.session_state.all_the_process += f"""
# Drop Rows
df.dropna(subset={fill_feat}, inplace=True)
df.reset_index(drop=True, inplace=True)
\n """
                    n_before = len(df)
                    df.dropna(subset=fill_feat, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    st.session_state['df'] = df
                    st.success(f"**{n_before - len(df)}** rows with missing values in **`{fill_feat}`** have been dropped. **{len(df)}** rows remain.")


                # Drop Columns (only the selected columns that actually have missing values)
                elif strategy == "Drop Columns":
                    drop_cols = [c for c in fill_feat if df[c].isnull().any()]
                    st.session_state.all_the_process += f"""
# Drop Columns
df.drop(columns={drop_cols}, inplace=True)
\n """
                    df.drop(columns=drop_cols, inplace=True)
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{drop_cols}`** have been dropped from the DataFrame.")
                    kept = [c for c in fill_feat if c not in drop_cols]
                    if kept:
                        st.info(f"The Columns **`{kept}`** have no missing values, so they were kept.")


                # Fill with Mean / Median (non-numeric features fall back to the mode)
                elif strategy in ("Fill with Mean", "Fill with Median"):
                    stat = "mean" if strategy == "Fill with Mean" else "median"
                    num_cols, cat_cols = fill_mean_median(df, fill_feat, stat)

                    st.session_state.all_the_process += f"""
# Fill with {stat.capitalize()} (numeric) and Mode (non-numeric)
df[{num_cols}] = df[{num_cols}].fillna(df[{num_cols}].{stat}())
for col in {cat_cols}:
    df[col] = df[col].fillna(df[col].mode().iloc[0])
\n """
                    st.session_state['df'] = df

                    if num_cols:
                        st.success(f"The numeric columns **`{num_cols}`** have been filled with the {stat}.")
                    if cat_cols:
                        st.info(f"The {stat} can't be computed for non-numeric columns, so **`{cat_cols}`** have been filled with the mode (most frequent value) instead.")


                # Fill with Mode
                elif strategy == "Fill with Mode (Most Frequent)":
                    st.session_state.all_the_process += f"""
# Fill with Mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])
\n """
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[fill_feat] = imputer.fit_transform(df[fill_feat])

                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")


                # Fill with ffill, bfill
                elif strategy == "Fill with ffill, bfill":
                    st.session_state.all_the_process += f"""
# Fill with ffill, bfill
df[{fill_feat}] = df[{fill_feat}].ffill().bfill()
\n """
                    df[fill_feat] = df[fill_feat].ffill().bfill()
                    st.session_state['df'] = df
                    st.success(f"The Columns **`{fill_feat}`** have been filled with ffill, bfill.")
        
        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([0.9, 0.6, 1])
        with col2:
            show_df = st.button("Show DataFrame", key="missing_show_df")
        if show_df:
            st.dataframe(df, use_container_width=True)
